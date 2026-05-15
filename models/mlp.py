"""
Simple Classifier for Classifier-Guided Diffusion

This is a simple CNN classifier used for classifier-guided diffusion.
It's trained separately and then used to guide the diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """
    Simple CNN classifier for MNIST.

    Used in classifier-guided diffusion to provide gradients
    that guide the denoising process towards a target class.
    """

    def __init__(self, num_classes=10, in_channels=1, time_embed_dim=64):
        """
        Args:
            num_classes: Number of output classes (10 for MNIST 0-9)
            in_channels: Number of input channels (1 for grayscale)
            time_embed_dim: Dimension of timestep embedding
        """
        super().__init__()

        # Timestep embedding: normalize t to [0,1] then project
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # FC input = flattened conv features + time embedding
        self.fc1 = nn.Linear(64 * 4 * 4 + time_embed_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, t):
        """
        Args:
            x: Noisy image [batch_size, channels, height, width]
            t: Timestep indices [batch_size]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Timestep embedding
        t_emb = self.time_embed(t.float().unsqueeze(1) / 1000.0)  # [batch, time_embed_dim]

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # [batch, 32, 14, 14]

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # [batch, 64, 7, 7]

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # [batch, 64, 4, 4]

        # Flatten and concatenate timestep embedding
        x = x.view(x.size(0), -1)  # [batch, 64*4*4]
        x = torch.cat([x, t_emb], dim=1)  # [batch, 64*4*4 + time_embed_dim]

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ConvNet(nn.Module):
    """
    Slightly more complex CNN classifier.

    Use this if SimpleClassifier isn't achieving good accuracy.
    """

    def __init__(self, num_classes=10, in_channels=1):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
        """
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fully connected
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input image [batch_size, channels, height, width]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def train_classifier(model, train_loader, num_epochs=5, learning_rate=1e-3, device='cuda'):
    """
    Train a classifier on MNIST.

    Args:
        model: Classifier model to train
        train_loader: DataLoader for training data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


if __name__ == "__main__":
    # Test the classifier
    model = SimpleClassifier(num_classes=10, in_channels=1)

    # Create dummy input
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images

    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test softmax probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {probs.sum(dim=1)}")
