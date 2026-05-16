"""
U-Net for DDPM
==============
The neural network that learns to predict noise ε given a noisy image x_t,
a timestep t, and an optional class label y.

Architecture:
    Encoder (downsample) → Bottleneck → Decoder (upsample) + skip connections

Conditioning:
    t and y are both embedded into a single 256-dim vector via sinusoidal
    encoding + nn.Embedding, then injected into every ConvBlock via AdaGN
    (Adaptive Group Normalisation — scale & shift learned from the embedding).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time Embedding ─────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """
    Converts a scalar timestep t into a rich embed_dim-dimensional vector
    using sinusoidal encoding (same as Transformer positional encoding).
    Nearby timesteps → nearby vectors, giving the model a smooth time sense.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]  →  output: [B, embed_dim]
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, embed_dim]


# ── Conv Block (ResBlock + AdaGN) ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Two-conv residual block with time/class conditioning via AdaGN.

    AdaGN: after GroupNorm, scale and shift are predicted from the
    time+class embedding rather than learned as fixed parameters.
    This lets every block adapt its behaviour to the current timestep.
    """

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1     = nn.GroupNorm(8, out_channels)
        # projects embedding → scale + shift (hence ×2)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)
        self.conv2     = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, out_channels)
        # 1×1 conv to align channels on the residual path when they differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))

        # AdaGN: modulate normalised features with embedding-predicted scale & shift
        scale, shift = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = F.gelu(h * (1 + scale) + shift)

        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ── U-Net ──────────────────────────────────────────────────────────────────────

class SimpleUNet(nn.Module):
    """
    Lightweight U-Net for 28×28 grayscale images.

    Input:  noisy image x_t  [B, 1, 28, 28]
            timestep t        [B]
            class label y     [B]  (use value 10 for "no class" / unconditional)
    Output: predicted noise ε [B, 1, 28, 28]

    The model outputs the NOISE that was added, not the clean image directly.
    This ε-parameterisation (Ho et al. 2020) gives more stable training.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: int = 64, time_embed_dim: int = 256,
                 num_classes: int = 10):
        super().__init__()
        self.null_token = num_classes   # label=10 means "unconditional"

        # ── Conditioning embeddings ──
        self.time_embedding  = TimeEmbedding(time_embed_dim)
        # indices 0–9: digit classes,  index 10: null token
        self.class_embedding = nn.Embedding(num_classes + 1, time_embed_dim)

        # ── Encoder ──
        self.down1 = ConvBlock(in_channels,  channels,     time_embed_dim)  # 28×28
        self.down2 = ConvBlock(channels,     channels * 2, time_embed_dim)  # 14×14

        # ── Bottleneck ──
        self.middle = ConvBlock(channels * 2, channels * 2, time_embed_dim) #  7×7

        # ── Decoder (channels ×4 / ×2 because skip connections are cat'd) ──
        self.up1 = ConvBlock(channels * 4, channels,     time_embed_dim)   # 14×14
        self.up2 = ConvBlock(channels * 2, channels,     time_embed_dim)   # 28×28

        # ── Output ──
        self.final = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor,
                t: torch.Tensor,
                class_label: torch.Tensor) -> torch.Tensor:
        # Build a single conditioning vector: time + class fused by addition
        cond = self.time_embedding(t) + self.class_embedding(class_label)

        # Encoder
        d1 = self.down1(x, cond)                                    # [B, C,  28, 28]
        d2 = self.down2(F.max_pool2d(d1, 2), cond)                  # [B, 2C, 14, 14]

        # Bottleneck
        m  = self.middle(F.max_pool2d(d2, 2), cond)                 # [B, 2C,  7,  7]

        # Decoder — upsample then concat skip connection
        u1 = self.up1(torch.cat([F.interpolate(m,  scale_factor=2), d2], dim=1), cond)
        u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2), d1], dim=1), cond)

        return self.final(u2)                                        # [B, 1, 28, 28]
