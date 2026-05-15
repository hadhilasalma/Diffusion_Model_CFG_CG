"""
U-Net Architecture for Diffusion Models

This module contains the U-Net model that learns to predict noise
in the reverse diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    Embed timestep into a vector (similar to positional encoding in transformers!)

    The model needs to know "which timestep are we at?" so it knows how noisy
    the image should be. This layer converts a scalar timestep into a vector.
    """

    def __init__(self, embed_dim):
        """
        Args:
            embed_dim: Dimension of the time embedding vector
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Convert timestep t to embedding vector using sinusoidal encoding.

        Args:
            t: Timestep tensor [batch_size]

        Returns:
            embedding: [batch_size, embed_dim]
        """
        # Create positional encoding (sin/cos pattern)
        device = t.device
        batch_size = t.shape[0]

        # Generate sinusoidal features
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Apply sin/cos (like transformers!)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)  # [batch_size, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch_size, embed_dim]

        return emb


class ConvBlock(nn.Module):
    """
    Two-conv ResBlock: (Conv -> GroupNorm -> AdaGN -> GELU) x2 + residual skip.

    Includes time embedding injection for conditioning.
    """

    def __init__(self, in_channels, out_channels, time_embed_dim):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embed_dim: Dimension of time embedding
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        # AdaGN: project conditioning to scale + shift (2× channels)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # 1×1 conv to match channels on the skip path when they differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        """
        Args:
            x: [batch_size, channels, height, width]
            time_emb: [batch_size, time_embed_dim]

        Returns:
            output: [batch_size, out_channels, height, width]
        """
        out = self.norm1(self.conv1(x))

        # AdaGN: split projection into scale and shift, apply to normalized features
        scale, shift = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        out = F.gelu(out * (1 + scale) + shift)

        out = F.gelu(self.norm2(self.conv2(out)))

        return out + self.skip(x)


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion model.

    Structure:
    - Downsampling path (squeeze information)
    - Middle layers (process at low resolution)
    - Upsampling path (expand back to original size)
    - Skip connections (preserve details)

    Key: This network learns to predict the noise that was added!
    """

    def __init__(self, in_channels=1, out_channels=1, channels=128, time_embed_dim=256, num_classes=10, null_token=None):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            out_channels: Number of output channels (same as input for noise prediction)
            channels: Base number of channels (will be scaled up in deeper layers)
            time_embed_dim: Dimension of time embedding
            num_classes: Number of classes for conditioning (10 for MNIST 0-9)
            null_token: Index used as unconditional token (defaults to num_classes)
        """
        super().__init__()

        if null_token is None:
            null_token = num_classes
        self.null_token = null_token
        self.num_classes = num_classes

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Class embedding (for conditional generation)
        # num_classes + 1: indices 0-9 are real classes, index 10 is null/unconditional token
        self.class_embedding = nn.Embedding(num_classes + 1, time_embed_dim)

        # Downsampling blocks (encoder)
        self.down1 = ConvBlock(in_channels, channels, time_embed_dim)
        self.down2 = ConvBlock(channels, channels * 2, time_embed_dim)

        # Middle blocks (bottleneck)
        self.middle = ConvBlock(channels * 2, channels * 2, time_embed_dim)

        # Upsampling blocks (decoder)
        self.up1 = ConvBlock(channels * 4, channels, time_embed_dim)  # *4 because of skip connection
        self.up2 = ConvBlock(channels * 2, channels, time_embed_dim)  # *2 because of skip connection

        # Final output
        self.final = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x, t, class_label=None):
        """
        Args:
            x: Noisy image [batch_size, channels, height, width]
            t: Timestep [batch_size]
            class_label: Class label [batch_size] or None for unconditional
                        Use value num_classes (10 for MNIST) for unconditional

        Returns:
            noise_prediction: Predicted noise [batch_size, out_channels, height, width]
        """
        # Validate input shapes
        assert x.dim() == 4, f"Expected x to have 4 dimensions, got {x.dim()}"
        assert t.dim() == 1, f"Expected t to have 1 dimension, got {t.dim()}"
        assert x.shape[0] == t.shape[0], f"Batch size mismatch: x has {x.shape[0]}, t has {t.shape[0]}"

        if class_label is not None:
            assert class_label.dim() == 1, f"Expected class_label to have 1 dimension, got {class_label.dim()}"
            assert class_label.shape[0] == x.shape[0], f"Batch size mismatch: x has {x.shape[0]}, class_label has {class_label.shape[0]}"
            assert (class_label >= 0).all() and (class_label <= 10).all(), "Class labels must be in range [0, 10]"

        batch_size = x.shape[0]

        # Get time embedding
        time_emb = self.time_embedding(t)  # [batch_size, time_embed_dim]

        # Add class information if provided (for guided generation)
        if class_label is not None:
            class_emb = self.class_embedding(class_label)  # [batch_size, time_embed_dim]
            time_emb = time_emb + class_emb

        # Downsampling with skip connections
        down1_out = self.down1(x, time_emb)
        down1_pool = F.max_pool2d(down1_out, 2)  # Reduce spatial size by 2x

        down2_out = self.down2(down1_pool, time_emb)
        down2_pool = F.max_pool2d(down2_out, 2)  # Reduce spatial size by 2x again

        # Middle processing (at lowest resolution)
        middle_out = self.middle(down2_pool, time_emb)

        # Upsampling with skip connections
        # Upsample and concatenate with skip connection from down2
        up1_input = torch.cat([F.interpolate(middle_out, scale_factor=2), down2_out], dim=1)
        up1_out = self.up1(up1_input, time_emb)

        # Upsample and concatenate with skip connection from down1
        up2_input = torch.cat([F.interpolate(up1_out, scale_factor=2), down1_out], dim=1)
        up2_out = self.up2(up2_input, time_emb)

        # Final prediction (noise)
        noise_pred = self.final(up2_out)

        return noise_pred


if __name__ == "__main__":
    # Test the model
    model = SimpleUNet(in_channels=1, out_channels=1, channels=64, time_embed_dim=128, num_classes=10)

    # Create dummy input
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 images
    t = torch.tensor([100, 500])    # Timesteps
    class_labels = torch.tensor([3, 7])  # Classes: 3 and 7

    # Forward pass
    output = model(x, t, class_labels)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
