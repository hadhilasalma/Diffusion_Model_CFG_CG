"""
Sampling script for the Diffusion Model

Generate images using a trained diffusion model.
Sampling logic lives in diffusion/sampling.py — this script
handles argument parsing, model loading, and saving results.

Usage:
  python sample.py --guidance-type unconditional
  python sample.py --guidance-type classifier-free  --guidance-scale 7.5
  python sample.py --guidance-type classifier-guided --cg-model-path checkpoints_cg/model_final.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import argparse

from models.unet import SimpleUNet
from models.classifier import SimpleClassifier
from diffusion.scheduler_ddpm import DDPMScheduler
from diffusion.sampling import sample_cfg, sample_cg


def denormalize(x):
    return (x + 1) / 2


def save_image_grid(images, filename, num_cols=10):
    num_images = images.shape[0]
    num_rows   = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_images:
                img = denormalize(images[idx].cpu().squeeze()).clamp(0, 1)
                axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from Diffusion Model')
    parser.add_argument('--model-path',     type=str,   default='checkpoints/model_final.pth',    help='Path to CFG-trained model')
    parser.add_argument('--cg-model-path',  type=str,   default='checkpoints_cg/model_final.pth', help='Path to unconditional model for CG')
    parser.add_argument('--classifier-path',type=str,   default='checkpoints/classifier_final.pth')
    parser.add_argument('--num-samples',    type=int,   default=100)
    parser.add_argument('--guidance-type',  type=str,   default='classifier-free',
                        choices=['classifier-free', 'classifier-guided', 'unconditional'])
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--device',         type=str,   default='cuda')
    parser.add_argument('--num-steps',      type=int,   default=1000)
    parser.add_argument('--output-dir',     type=str,   default='results')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SAMPLING FROM DIFFUSION MODEL")
    print("=" * 80)

    # Load the correct U-Net for the chosen guidance type
    model_path = args.cg_model_path if args.guidance_type == 'classifier-guided' else args.model_path
    print(f"\nLoading model from {model_path}...")
    model      = SimpleUNet(in_channels=1, out_channels=1, channels=64, num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
    model = model.to(device)

    scheduler = DDPMScheduler(num_timesteps=1000)
    scheduler.to(device)

    # ── Unconditional ─────────────────────────────────────────────────────────
    if args.guidance_type == 'unconditional':
        print("\nGenerating unconditional samples...")
        samples = sample_cfg(model, scheduler, num_samples=args.num_samples,
                             class_labels=None, guidance_scale=1.0,
                             device=device, num_steps=args.num_steps)
        save_image_grid(samples, os.path.join(args.output_dir, 'samples_unconditional.png'))

    # ── Classifier-Free Guidance ──────────────────────────────────────────────
    elif args.guidance_type == 'classifier-free':
        print("\nGenerating samples with classifier-free guidance...")
        per_class = args.num_samples // 10

        for class_id in range(10):
            class_labels = torch.full((per_class,), class_id, dtype=torch.long, device=device)
            samples = sample_cfg(model, scheduler, num_samples=per_class,
                                 class_labels=class_labels, guidance_scale=args.guidance_scale,
                                 device=device, num_steps=args.num_steps)
            save_image_grid(samples, os.path.join(args.output_dir,
                            f'samples_cfg_class_{class_id}_scale_{args.guidance_scale}.png'))

        print("\nGenerating across guidance scales (digit 3)...")
        class_labels = torch.full((args.num_samples,), 3, dtype=torch.long, device=device)
        for scale in [1.0, 3.0, 7.5, 15.0]:
            samples = sample_cfg(model, scheduler, num_samples=args.num_samples,
                                 class_labels=class_labels, guidance_scale=scale,
                                 device=device, num_steps=args.num_steps)
            save_image_grid(samples, os.path.join(args.output_dir,
                            f'samples_cfg_scale_{scale}.png'))

    # ── Classifier-Guided ─────────────────────────────────────────────────────
    elif args.guidance_type == 'classifier-guided':
        print(f"\nLoading classifier from {args.classifier_path}...")
        classifier = SimpleClassifier(num_classes=10)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
        classifier = classifier.to(device)

        print("\nGenerating samples with classifier-guided diffusion...")
        per_class = args.num_samples // 10
        for class_id in range(10):
            class_labels = torch.full((per_class,), class_id, dtype=torch.long, device=device)
            samples = sample_cg(model, classifier, scheduler, num_samples=per_class,
                                class_labels=class_labels, guidance_scale=args.guidance_scale,
                                device=device, num_steps=args.num_steps)
            save_image_grid(samples, os.path.join(args.output_dir,
                            f'samples_cg_class_{class_id}.png'))

        print("\nGenerating across guidance scales (digit 3)...")
        class_labels = torch.full((args.num_samples,), 3, dtype=torch.long, device=device)
        for scale in [0.1, 0.5, 1.0, 2.0]:
            samples = sample_cg(model, classifier, scheduler, num_samples=args.num_samples,
                                class_labels=class_labels, guidance_scale=scale,
                                device=device, num_steps=args.num_steps)
            save_image_grid(samples, os.path.join(args.output_dir,
                            f'samples_cg_scale_{scale}.png'))

    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
