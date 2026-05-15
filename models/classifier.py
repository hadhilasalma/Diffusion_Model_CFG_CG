import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import TimeEmbedding


class ResBlock(nn.Module):
    """
    Residual block with timestep embedding injection via AdaGN.

    Time embedding is projected and added after the first conv so the
    block knows which noise level it is operating at — critical for
    producing reliable gradients across all diffusion timesteps.
    """

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()

        self.conv1    = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1      = nn.GroupNorm(8, out_channels)

        self.conv2    = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2      = nn.GroupNorm(8, out_channels)

        # AdaGN: project conditioning to scale + shift (2× channels)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)

        # 1×1 conv to match channels on the skip path when they differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.gn1(self.conv1(x))

        # AdaGN: modulate normalized features with scale + shift before activation
        scale, shift = self.time_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(h * (1 + scale) + shift)

        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over spatial positions.

    Matches the AttentionBlock in Dhariwal & Nichol 2021:
      GroupNorm → QKV (1×1 conv) → scaled dot-product attention
      → output projection (1×1 conv) → residual add.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm     = nn.GroupNorm(8, channels)
        self.qkv      = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)          # each [B, C, H, W]

        head_dim = C // self.num_heads
        # [B, num_heads, head_dim, H*W]
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        # Scaled dot-product attention — Dhariwal & Nichol 2021 §A
        scale = head_dim ** -0.5
        attn  = torch.einsum('bndi,bndj->bnij', q, k) * scale  # [B, heads, HW, HW]
        attn  = F.softmax(attn, dim=-1)

        out = torch.einsum('bnij,bndj->bndi', attn, v)          # [B, heads, head_dim, HW]
        out = out.reshape(B, C, H, W)
        return x + self.proj_out(out)


class SimpleClassifier(nn.Module):
    """
    Noise-aware classifier for MNIST used in classifier-guided (CG) diffusion.

    Architecture matches the downsampling stack of the ADM classifier
    (Dhariwal & Nichol 2021):
        InitConv → ResBlock → AvgPool → ResBlock → AvgPool
        → SpatialAttention → AdaptiveAvgPool → FC → FC
    """

    def __init__(self, num_classes=10, in_channels=1, time_embed_dim=64, num_heads=1):
        super().__init__()

        self.time_embed = TimeEmbedding(time_embed_dim)

        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        self.res1  = ResBlock(32,  64,  time_embed_dim)
        # AvgPool2d for downsampling — matches ADM EncoderUNetModel (Dhariwal & Nichol 2021)
        self.down1 = nn.AvgPool2d(2, stride=2)          # 28×28 → 14×14

        self.res2  = ResBlock(64, 128, time_embed_dim)
        self.down2 = nn.AvgPool2d(2, stride=2)          # 14×14 →  7×7

        # Spatial (QKV) attention after final ResBlock
        self.attn = SpatialAttention(128, num_heads=num_heads)

        # Pool to fixed spatial size before FC
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, t):
        """
        Args:
            x: Noisy image  [batch, channels, H, W]
            t: Timestep     [batch]
        Returns:
            logits          [batch, num_classes]
        """
        t_emb = self.time_embed(t)                      # [B, time_embed_dim]

        x = F.silu(self.init_conv(x))                   # [B,  32, 28, 28]

        x = self.res1(x, t_emb)                         # [B,  64, 28, 28]
        x = self.down1(x)                               # [B,  64, 14, 14]

        x = self.res2(x, t_emb)                         # [B, 128, 14, 14]
        x = self.down2(x)                               # [B, 128,  7,  7]

        x = self.attn(x)                                # [B, 128,  7,  7]

        x = self.pool(x)                                # [B, 128,  4,  4]
        x = x.view(x.size(0), -1)                       # [B, 2048]

        x = F.silu(self.fc1(x))
        return self.fc2(x)
