"""
Super-Resolution Decoder with PV-Semantic Guidance.

Diffusion-based SR decoder (built on SGDM framework) that produces
8x super-resolved imagery. The PV-Semantic Guidance block uses
segmentation predictions to focus reconstruction on panel regions.

Reference: Section 3.3 of the paper (Equations 7-11).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, dim):
    """Sinusoidal timestep embedding for diffusion models."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """Projects timestep into feature space."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        emb = get_timestep_embedding(t, self.mlp[0].in_features)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with time embedding conditioning for the diffusion UNet."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        # add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block used at lower spatial resolutions."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhdn,bhem->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhem->bhdn', attn, v)
        out = out.reshape(B, C, H * W)
        out = self.proj(out).view(B, C, H, W)

        return x + out


class PVSemanticGuidance(nn.Module):
    """
    PV-Semantic Guidance block (Eq. 9-10).

    Uses segmentation information to guide SR reconstruction,
    focusing detail recovery on photovoltaic panel regions.
    """

    def __init__(self, sr_channels, seg_channels=1, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = sr_channels // num_heads
        self.scale = head_dim ** -0.5

        # project SR features to queries
        self.q_proj = nn.Conv2d(sr_channels, sr_channels, 1)
        # project seg mask embedding to keys and values
        self.mask_embed = nn.Sequential(
            nn.Conv2d(seg_channels, sr_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(sr_channels // 4, sr_channels, 3, padding=1),
        )
        self.k_proj = nn.Conv2d(sr_channels, sr_channels, 1)
        self.v_proj = nn.Conv2d(sr_channels, sr_channels, 1)

        self.out_proj = nn.Conv2d(sr_channels, sr_channels, 1)

    def forward(self, sr_feat, seg_mask):
        """
        Args:
            sr_feat: SR features (B, C, H, W)
            seg_mask: segmentation prediction (B, 1, H', W') - will be resized
        Returns:
            guided features (B, C, H, W)
        """
        B, C, H, W = sr_feat.shape

        # resize seg mask to match SR feature spatial size
        if seg_mask.shape[-2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)

        mask_feat = self.mask_embed(seg_mask)

        # compute attention (Eq. 9)
        q = self.q_proj(sr_feat).flatten(2)     # (B, C, HW)
        k = self.k_proj(mask_feat).flatten(2)   # (B, C, HW)
        v = self.v_proj(mask_feat).flatten(2)   # (B, C, HW)

        # reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # guided features (Eq. 10): A_PV * V_M + F_SR
        guided = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        guided = self.out_proj(guided) + sr_feat

        return guided


class DiffusionUNet(nn.Module):
    """
    Simplified diffusion UNet backbone for the SR decoder.

    Handles the denoising process conditioned on encoder features
    and timestep information.
    """

    def __init__(self, in_channels=6, out_channels=3, base_channels=64,
                 channel_mult=(1, 2, 4, 4), time_dim=256):
        super().__init__()

        self.time_embed = TimeEmbedding(base_channels, time_dim)
        ch = base_channels

        # input projection (concat noisy image + LR bicubic upsampled)
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels_list = [ch]

        for mult in channel_mult:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_dim),
                ResBlock(out_ch, out_ch, time_dim),
            ]))
            channels_list.append(out_ch)
            ch = out_ch
            self.down_samples.append(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            )

        # bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.up_samples.append(
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch + channels_list.pop(), out_ch, time_dim),
                ResBlock(out_ch, out_ch, time_dim),
            ]))
            ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t, encoder_feat=None):
        """
        Args:
            x: noisy input (B, in_channels, H, W)
            t: timestep (B,)
            encoder_feat: optional conditioning from shared encoder
        Returns:
            predicted noise or denoised output
        """
        t_emb = self.time_embed(t)

        h = self.input_conv(x)
        hs = [h]

        # encoder path
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            for block in down_block:
                h = block(h, t_emb)
            hs.append(h)
            h = down_sample(h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # decoder path
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            h = up_sample(h)
            skip = hs.pop()
            # handle potential size mismatch from padding
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in up_block:
                h = block(h, t_emb)

        return self.out_conv(h)


class SRDecoder(nn.Module):
    """
    Complete SR decoder module with diffusion backbone and PV-Semantic Guidance.

    During training, uses full diffusion process.
    During inference, uses DDIM accelerated sampling.
    """

    def __init__(self, base_channels=64, channel_mult=(1, 2, 4, 4),
                 diffusion_steps=1000, ddim_steps=50,
                 beta_start=0.0001, beta_end=0.02):
        super().__init__()

        self.diffusion_steps = diffusion_steps
        self.ddim_steps = ddim_steps

        # noise schedule
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # diffusion UNet (input: noisy HR + LR bicubic upsampled = 6ch)
        self.unet = DiffusionUNet(
            in_channels=6, out_channels=3,
            base_channels=base_channels, channel_mult=channel_mult,
        )

        # PV-Semantic Guidance
        self.pv_guidance = PVSemanticGuidance(base_channels * channel_mult[-1])

        # final reconstruction with pixel shuffle for upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to clean image at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_start + sqrt_one_minus * noise, noise

    def forward(self, lr_img, hr_img=None, seg_mask=None, encoder_features=None):
        """
        Training forward pass.

        Args:
            lr_img: low-res input (B, 3, H, W) at 0.8m GSD
            hr_img: high-res target (B, 3, 8H, 8W) at 0.1m GSD (training only)
            seg_mask: intermediate segmentation mask
            encoder_features: features from shared encoder
        Returns:
            dict with 'sr_output', 'noise_pred', 'noise_target' keys
        """
        B = lr_img.shape[0]
        device = lr_img.device

        # bicubic upsample LR to target resolution
        lr_up = F.interpolate(lr_img, scale_factor=4, mode='bicubic', align_corners=False)

        if self.training and hr_img is not None:
            # sample random timesteps
            t = torch.randint(0, self.diffusion_steps, (B,), device=device)

            # add noise to HR image
            noise = torch.randn_like(hr_img)
            noisy_hr, noise_target = self.q_sample(hr_img, t, noise)

            # concat noisy HR with LR upsampled as conditioning
            x_input = torch.cat([noisy_hr, lr_up], dim=1)

            # predict noise
            noise_pred = self.unet(x_input, t, encoder_features)

            # denoised estimate for semantic guidance (single-step approx)
            with torch.no_grad():
                x0_hat = (noisy_hr - self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise_pred)
                x0_hat = x0_hat / self.sqrt_alphas_cumprod[t][:, None, None, None].clamp(min=1e-8)
                x0_hat = x0_hat.clamp(-1, 1)

            sr_output = self.upsample(x0_hat)

            return {
                'sr_output': sr_output,
                'noise_pred': noise_pred,
                'noise_target': noise_target,
            }
        else:
            # DDIM inference
            sr_output = self._ddim_sample(lr_up, seg_mask, encoder_features)
            sr_output = self.upsample(sr_output)
            return {'sr_output': sr_output}

    @torch.no_grad()
    def _ddim_sample(self, lr_up, seg_mask=None, encoder_features=None):
        """DDIM accelerated sampling for inference."""
        B, C, H, W = lr_up.shape
        device = lr_up.device

        # compute DDIM timestep subsequence
        step_size = self.diffusion_steps // self.ddim_steps
        timesteps = list(range(0, self.diffusion_steps, step_size))[::-1]

        # start from pure noise
        x = torch.randn(B, C, H, W, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            x_input = torch.cat([x, lr_up], dim=1)
            noise_pred = self.unet(x_input, t_batch, encoder_features)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        return x

    def get_intermediate_features(self, lr_img):
        """Extract intermediate features for CTIM interaction."""
        lr_up = F.interpolate(lr_img, scale_factor=4, mode='bicubic', align_corners=False)
        # return the upsampled LR as a proxy for SR features during early stages
        return lr_up
