"""
Noise-Aware Classifier for Classifier Guidance (CG)
====================================================
Used ONLY during CG inference — not during diffusion model training.

Why noise-aware? A standard classifier trained on clean images gives useless
gradients when applied to a heavily noisy x_t. This classifier is trained on
noisy images across all timesteps 0–999, so it can classify at any noise level.

Architecture mirrors the ADM classifier (Dhariwal & Nichol 2021):
    Conv → ResBlock → AvgPool → ResBlock → AvgPool → SpatialAttention → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import TimeEmbedding


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual conv block with AdaGN time conditioning.
    Identical in design to ConvBlock in unet.py — keeps the classifier
    aware of the current noise level at every layer.
    """

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1       = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)  # → scale + shift
        self.conv2     = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2       = nn.GroupNorm(8, out_channels)
        self.skip      = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.gn1(self.conv1(x))
        scale, shift = self.time_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(h * (1 + scale) + shift)
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over spatial positions.
    Lets each pixel attend to all others — important for global digit structure
    (e.g. "this loop at top + this stroke at bottom = digit 9").
    Follows the AttentionBlock in Dhariwal & Nichol 2021.
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.norm      = nn.GroupNorm(8, channels)
        self.qkv       = nn.Conv2d(channels, channels * 3, 1)   # Q, K, V in one shot
        self.proj_out  = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        attn = torch.einsum("bndi,bndj->bnij", q, k) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum("bnij,bndj->bndi", attn, v).reshape(B, C, H, W)
        return x + self.proj_out(out)


# ── Classifier ─────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    """
    Noise-aware MNIST classifier.

    Takes (noisy image x_t, timestep t) and returns class logits [B, 10].
    The timestep is embedded via the same sinusoidal encoding as the U-Net
    and injected into every ResBlock via AdaGN.

    Input:  x_t  [B, 1, 28, 28],  t [B]
    Output: logits [B, num_classes]
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1,
                 time_embed_dim: int = 64, num_heads: int = 1):
        super().__init__()

        # ── Conditioning ──
        self.time_embed = TimeEmbedding(time_embed_dim)

        # ── Encoder ──
        self.init_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.res1      = ResBlock(32,  64,  time_embed_dim)
        self.down1     = nn.AvgPool2d(2)   # 28×28 → 14×14
        self.res2      = ResBlock(64, 128, time_embed_dim)
        self.down2     = nn.AvgPool2d(2)   # 14×14 →  7×7

        # ── Global context ──
        self.attn = SpatialAttention(128, num_heads)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Classifier head ──
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        x = F.silu(self.init_conv(x))    # [B,  32, 28, 28]
        x = self.down1(self.res1(x, t_emb))  # [B,  64, 14, 14]
        x = self.down2(self.res2(x, t_emb))  # [B, 128,  7,  7]
        x = self.attn(x)                 # [B, 128,  7,  7]
        x = self.pool(x).flatten(1)      # [B, 2048]
        return self.fc2(F.silu(self.fc1(x)))
