"""
Training script for CG Diffusion Model (Unconditional)

Trains a purely unconditional U-Net on MNIST — every sample always
receives the null token (label=10), with no class conditioning at all.
This gives the model a clean unconditional denoising objective used
by the classifier-guided (CG) sampler at inference time.

Shared training loop lives in diffusion/training.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from models.unet import SimpleUNet
from diffusion.scheduler_ddpm import DDPMScheduler
from diffusion.training import train_diffusion


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Train Unconditional Diffusion Model for CG')
    parser.add_argument('--epochs',         type=int,   default=20,              help='Number of epochs')
    parser.add_argument('--batch-size',     type=int,   default=64,              help='Batch size')
    parser.add_argument('--learning-rate',  type=float, default=1e-4,            help='Learning rate')
    parser.add_argument('--device',         type=str,   default='cuda',          help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str,   default='checkpoints_cg',help='Checkpoint directory')
    parser.add_argument('--model-channels', type=int,   default=64,              help='Base channels for U-Net')
    args = parser.parse_args()

    print("=" * 80)
    print("DIFFUSION MODEL TRAINING  (CG — Unconditional)")
    print("=" * 80)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {total_memory:.1f} GB")

    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("\nCreating U-Net model...")
    model = SimpleUNet(in_channels=1, out_channels=1, channels=args.model_channels,
                       time_embed_dim=256, num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = DDPMScheduler(num_timesteps=1000)

    print("\nStarting training...")
    model, optimizer, losses, val_losses = train_diffusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        mode='cg',
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    final_path = os.path.join(args.checkpoint_dir, 'model_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': losses,
        'scheduler': {
            'betas': scheduler.betas, 'alphas': scheduler.alphas,
            'alphas_cumprod': scheduler.alphas_cumprod,
            'num_timesteps': scheduler.num_timesteps,
        },
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    try:
        epochs_range = range(1, len(losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, losses, marker='o', label='Train Loss')
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training & Validation Loss - Unconditional Diffusion Model (CG)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss_cg.png', dpi=150, bbox_inches='tight')
        print("Loss curve saved: training_loss_cg.png")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("Use this model for classifier-guided sampling:")
    print("  python sample.py --guidance-type classifier-guided --cg-model-path checkpoints_cg/model_final.pth")
    print("=" * 80)


if __name__ == "__main__":
    main()
