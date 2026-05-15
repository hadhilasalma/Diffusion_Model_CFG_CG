import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTFeatureExtractor(nn.Module):
    """
    Clean-image CNN for domain-specific FID computation on MNIST.

    Unlike SimpleClassifier, this network:
      - Takes only the image (no timestep / noise conditioning)
      - Is trained exclusively on clean MNIST images
      - Produces a 256-dim feature vector from its penultimate layer

    The 256-dim features are passed to the Fréchet distance formula
    instead of raw pixels, giving a more semantically meaningful FID.
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1×28×28 → 32×14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 28 → 14

            # Block 2: 32×14×14 → 64×7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 14 → 7

            # Block 3: 64×7×7 → 128×3×3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),  # → 3×3 regardless of input size
        )

        self.embedding = nn.Sequential(
            nn.Linear(128 * 3 * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-dim feature vector — used for FID computation."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits — used during training."""
        return self.classifier(self.extract_features(x))
