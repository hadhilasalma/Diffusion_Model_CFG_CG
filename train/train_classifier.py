"""
Training script for the noise-aware classifier used in CG diffusion.

Trains SimpleClassifier on noisy MNIST images across all timesteps so it
can produce reliable gradients during classifier-guided sampling.
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

from models.classifier import SimpleClassifier
from diffusion.scheduler_ddpm import DDPMScheduler


def main():
    parser = argparse.ArgumentParser(description='Train Classifier for CG Guidance')
    parser.add_argument('--epochs',         type=int,   default=20,           help='Number of epochs')
    parser.add_argument('--batch-size',     type=int,   default=64,           help='Batch size')
    parser.add_argument('--learning-rate',  type=float, default=0.001,        help='Learning rate')
    parser.add_argument('--device',         type=str,   default='cuda',       help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str,   default='checkpoints',help='Checkpoint directory')
    args = parser.parse_args()

    print("=" * 80)
    print("CLASSIFIER TRAINING  (for CG Guidance)")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    scheduler = DDPMScheduler(num_timesteps=1000)
    scheduler.to(device)

    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = MNIST(root='./data', train=True,  transform=transform, download=True)
    test_dataset  = MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)
    print(f"Training samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

    print("\nCreating classifier...")
    classifier = SimpleClassifier(num_classes=10, in_channels=1)
    classifier  = classifier.to(device)
    print(f"Model parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        # ── Train ────────────────────────────────────────────────────────────
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            t            = torch.randint(0, 1000, (images.shape[0],), device=device)
            noisy_images = scheduler.forward_process(images, t, torch.randn_like(images))

            logits = classifier(noisy_images, t)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': loss.item()})

        train_loss     = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # ── Test (with per-noise-level breakdown) ────────────────────────────
        classifier.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        noise_level_stats = {}

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                t            = torch.randint(0, 1000, (images.shape[0],), device=device)
                noisy_images = scheduler.forward_process(images, t, torch.randn_like(images))

                logits = classifier(noisy_images, t)
                loss   = criterion(logits, labels)

                test_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                test_total   += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                for i, timestep in enumerate(t):
                    t_val = timestep.item()
                    if   t_val < 250: level = "Low Noise   (0–250)"
                    elif t_val < 500: level = "Medium Noise(250–500)"
                    elif t_val < 750: level = "High Noise  (500–750)"
                    else:             level = "Very High   (750–1000)"

                    if level not in noise_level_stats:
                        noise_level_stats[level] = {'correct': 0, 'total': 0}
                    noise_level_stats[level]['total'] += 1
                    if predicted[i] == labels[i]:
                        noise_level_stats[level]['correct'] += 1

        test_accuracy = 100 * test_correct / test_total

        print(f"\nEpoch {epoch+1}")
        print(f"  Train — Loss: {train_loss:.4f}  Accuracy: {train_accuracy:.2f}%")
        print(f"  Test  — Loss: {test_loss/len(test_loader):.4f}  Accuracy: {test_accuracy:.2f}%")
        print("  Accuracy by noise level:")
        for level in sorted(noise_level_stats.keys()):
            s   = noise_level_stats[level]
            acc = 100 * s['correct'] / s['total']
            print(f"    {level}: {acc:.2f}%  ({s['correct']}/{s['total']})")

        if test_accuracy > best_accuracy:
            best_accuracy  = test_accuracy
            best_path      = os.path.join(args.checkpoint_dir, 'classifier_best.pth')
            torch.save(classifier.state_dict(), best_path)
            print(f"  Best model saved: {best_path}")

    final_path = os.path.join(args.checkpoint_dir, 'classifier_final.pth')
    torch.save(classifier.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("\nRun classifier-guided sampling with:")
    print("  python sample.py --guidance-type classifier-guided")
    print("=" * 80)


if __name__ == "__main__":
    main()
