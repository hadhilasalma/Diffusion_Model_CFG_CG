"""
Training script for the clean-image feature extractor used in domain-specific FID.

Trains MNISTFeatureExtractor on clean MNIST images only — no noise, no timestep.
The trained model is saved and loaded by evaluate.py to compute a more
semantically meaningful FID than raw-pixel comparison.

Usage:
  python train_fid_extractor.py
  python train_fid_extractor.py --epochs 15 --checkpoint-dir checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from models.fid_extractor import MNISTFeatureExtractor


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train FID Feature Extractor')
    parser.add_argument('--epochs',         type=int,   default=10)
    parser.add_argument('--batch-size',     type=int,   default=128)
    parser.add_argument('--learning-rate',  type=float, default=1e-3)
    parser.add_argument('--feature-dim',    type=int,   default=256)
    parser.add_argument('--checkpoint-dir', type=str,   default='checkpoints')
    parser.add_argument('--device',         type=str,   default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("FID FEATURE EXTRACTOR TRAINING  (clean MNIST, no noise)")
    print("=" * 70)
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Feature dim : {args.feature_dim}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = MNIST('./data', train=True,  transform=transform, download=True)
    test_dataset  = MNIST('./data', train=False, transform=transform, download=True)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_dataset):,}  |  Test: {len(test_dataset):,}")

    model     = MNISTFeatureExtractor(feature_dim=args.feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    best_accuracy = 0.0
    save_path     = os.path.join(args.checkpoint_dir, 'fid_extractor.pth')

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"  Train — Loss: {total_loss/len(train_loader):.4f}  Acc: {train_acc:.2f}%")
        print(f"  Test  — Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model ({best_accuracy:.2f}%) → {save_path}")

    print("\n" + "=" * 70)
    print(f"Training complete.  Best test accuracy: {best_accuracy:.2f}%")
    print(f"Feature extractor saved to: {save_path}")
    print("\nNow run evaluation with domain-specific FID:")
    print("  python evaluate.py --guidance-type both")
    print("=" * 70)


if __name__ == "__main__":
    main()
