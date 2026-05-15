"""
Evaluation script for Diffusion Model

Computes quantitative metrics and visual comparisons for CFG and/or CG:
  - FID Score          (pixel-based, lower is better)
  - Inception Score    (MNIST classifier, higher is better)
  - Class Accuracy     (% of generated samples classified correctly)
  - Comparison grid    (Real vs CFG vs CG side-by-side)
  - Forward diffusion  (noising visualization)

Usage:
  python evaluate.py --guidance-type classifier-free
  python evaluate.py --guidance-type classifier-guided
  python evaluate.py --guidance-type both
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from models.unet import SimpleUNet
from models.classifier import SimpleClassifier
from models.fid_extractor import MNISTFeatureExtractor
from diffusion.scheduler_ddpm import DDPMScheduler
from diffusion.sampling import sample_cfg, sample_cg


# ── Utilities ─────────────────────────────────────────────────────────────────

def denormalize(x):
    return (x + 1) / 2


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_fid(real_images, gen_images):
    """
    Simplified pixel-based FID score.

    NOTE: True FID uses Inception-v3 features. This version uses raw pixels
    and is only suitable for relative comparison within this project.
    Formula: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r · Σ_g))
    """
    def stats(images):
        flat = images.view(images.shape[0], -1).numpy()
        mu   = flat.mean(axis=0)
        cov  = np.cov(flat, rowvar=False)
        return mu, cov

    mu_r, cov_r = stats(real_images)
    mu_g, cov_g = stats(gen_images)

    try:
        diff    = mu_r - mu_g
        covmean = scipy.linalg.sqrtm(cov_r @ cov_g)
        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)
        fid = float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))
    except Exception as e:
        print(f"  FID computation error: {e}")
        fid = float('inf')

    return fid


def compute_fid_domain(real_images, gen_images, extractor, device):
    """
    Domain-specific FID using a clean-image MNIST feature extractor.

    Extracts 256-dim features from both real and generated images via
    MNISTFeatureExtractor (trained on clean MNIST, no noise conditioning),
    then applies the same Fréchet distance formula as compute_fid().
    This gives a more semantically meaningful score than raw pixels.
    """
    extractor.eval()

    def extract(images):
        feats = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device)
                feats.append(extractor.extract_features(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    feat_r = extract(real_images)
    feat_g = extract(gen_images)

    mu_r, cov_r = feat_r.mean(axis=0), np.cov(feat_r, rowvar=False)
    mu_g, cov_g = feat_g.mean(axis=0), np.cov(feat_g, rowvar=False)

    try:
        diff    = mu_r - mu_g
        covmean = scipy.linalg.sqrtm(cov_r @ cov_g)
        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)
        fid = float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))
    except Exception as e:
        print(f"  Domain FID computation error: {e}")
        fid = float('inf')

    return fid


def compute_inception_score(logits, num_splits=10):
    """
    Inception Score from classifier logits.
    IS = exp( E[ KL( p(y|x) || p(y) ) ] )
    Higher is better. Uses MNIST classifier — not comparable to ImageNet IS.
    """
    probs = F.softmax(logits, dim=1)
    n     = probs.shape[0]
    num_splits = max(1, min(num_splits, n // 2))
    split_size = n // num_splits

    scores = []
    for i in range(num_splits):
        start   = i * split_size
        end     = (i + 1) * split_size if i < num_splits - 1 else n
        p_yx    = torch.clamp(probs[start:end], min=1e-10)
        p_y     = torch.clamp(p_yx.mean(dim=0), min=1e-10)
        kl      = (p_yx * (torch.log(p_yx) - torch.log(p_y))).sum(dim=1)
        scores.append(torch.exp(kl.mean()).item())

    return float(np.exp(np.mean(np.log(scores))))


def compute_class_accuracy(logits, class_labels):
    """% of generated samples whose predicted class matches the intended class."""
    predicted = torch.argmax(logits, dim=1)
    return 100.0 * (predicted == class_labels).sum().item() / len(class_labels)


# ── Visuals ───────────────────────────────────────────────────────────────────

def visualize_diffusion_process(x0, scheduler, output_dir, device):
    """Visualize forward noising at key timesteps."""
    timesteps = [0, 100, 250, 500, 750, 999]
    _, axes = plt.subplots(1, len(timesteps), figsize=(len(timesteps) * 2, 2))

    x0    = x0.to(device)
    noise = torch.randn_like(x0)

    for i, t in enumerate(timesteps):
        t_batch = torch.tensor([t], device=device)
        x_t     = scheduler.forward_process(x0, t_batch, noise)
        img     = denormalize(x_t[0].cpu().squeeze()).clamp(0, 1)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f't={t}')
        axes[i].axis('off')

    plt.suptitle('Forward Diffusion Process (Noising)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'diffusion_process.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_comparison_grid(real, cfg_samples, cg_samples, output_dir, n=10):
    """Save Real | CFG | CG side-by-side grid."""
    rows   = ['Real', 'CFG Generated', 'CG Generated']
    images = [real, cfg_samples, cg_samples]

    # Drop CG row if not available
    rows   = [r for r, img in zip(rows, images) if img is not None]
    images = [img for img in images if img is not None]

    n   = min(n, images[0].shape[0])
    _, axes = plt.subplots(len(rows), n, figsize=(n * 2, len(rows) * 2))
    if len(rows) == 1:
        axes = axes[None, :]

    for r, (label, imgs) in enumerate(zip(rows, images)):
        for c in range(n):
            img = denormalize(imgs[c].cpu().squeeze()).clamp(0, 1)
            axes[r, c].imshow(img, cmap='gray')
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=40)

    plt.suptitle('Real vs CFG vs CG', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_per_class_grid(samples, title, filename, n_per_class=5):
    """Save a grid with n_per_class samples per digit (0–9)."""
    _, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 2, 20))

    for class_id in range(10):
        start = class_id * n_per_class
        for j in range(n_per_class):
            img = denormalize(samples[start + j].cpu().squeeze()).clamp(0, 1)
            axes[class_id, j].imshow(img, cmap='gray')
            axes[class_id, j].axis('off')
        axes[class_id, 0].set_ylabel(f'Digit {class_id}', fontsize=9,
                                     rotation=90, labelpad=35)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ── Core evaluation ───────────────────────────────────────────────────────────

def generate_and_evaluate(
    guidance_type, model, classifier, scheduler,
    num_samples, num_steps, guidance_scale, device,
    cg_classifier=None,
    eval_classifier=None,
):
    """
    Generate samples for all 10 classes and compute FID / IS / class accuracy.
    Returns dict with metrics + generated samples tensor.

    eval_classifier: if provided, used for IS and class accuracy (clean CNN,
                     no noise conditioning). Falls back to noise-aware classifier
                     when None.
    """
    per_class   = num_samples // 10
    all_samples = []
    all_labels  = []

    print(f"\n  Generating {num_samples} samples ({per_class} per class)...")

    for class_id in range(10):
        labels = torch.full((per_class,), class_id, dtype=torch.long, device=device)

        if guidance_type == 'classifier-free':
            samples = sample_cfg(
                model, scheduler, num_samples=per_class,
                class_labels=labels, guidance_scale=guidance_scale,
                device=device, num_steps=num_steps,
            )
        else:
            samples = sample_cg(
                model, cg_classifier, scheduler, num_samples=per_class,
                class_labels=labels, guidance_scale=guidance_scale,
                device=device, num_steps=num_steps,
            )

        all_samples.append(samples.cpu())
        all_labels.extend([class_id] * per_class)

    all_samples = torch.cat(all_samples, dim=0)           # [N, 1, 28, 28]
    all_labels  = torch.tensor(all_labels, dtype=torch.long)

    # Use clean classifier for IS/accuracy if available; fall back to noise-aware
    if eval_classifier is not None:
        print("  IS/Accuracy judge: clean MNISTFeatureExtractor")
        eval_classifier.eval()
        with torch.no_grad():
            logits = eval_classifier(all_samples.to(device)).cpu()
    else:
        print("  IS/Accuracy judge: noise-aware classifier (fallback, t=0)")
        t_zero = torch.zeros(all_samples.shape[0], dtype=torch.long, device=device)
        classifier.eval()
        with torch.no_grad():
            logits = classifier(all_samples.to(device), t_zero).cpu()

    return {
        'samples':          all_samples,
        'labels':           all_labels,
        'inception_score':  compute_inception_score(logits),
        'class_accuracy':   compute_class_accuracy(logits, all_labels),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Model')
    parser.add_argument('--guidance-type',   type=str,   default='classifier-free',
                        choices=['classifier-free', 'classifier-guided', 'both'])
    parser.add_argument('--model-path',      type=str,   default='checkpoints/model_final.pth',
                        help='CFG U-Net checkpoint')
    parser.add_argument('--cg-model-path',   type=str,   default='checkpoints_cg/model_final.pth',
                        help='CG unconditional U-Net checkpoint')
    parser.add_argument('--classifier-path', type=str,   default='checkpoints/classifier_final.pth',
                        help='Noise-aware classifier checkpoint')
    parser.add_argument('--num-samples',     type=int,   default=100,
                        help='Total samples (must be divisible by 10)')
    parser.add_argument('--num-steps',       type=int,   default=1000,
                        help='Denoising steps (1000 for DDPM)')
    parser.add_argument('--cfg-scale',       type=float, default=7.5,
                        help='CFG guidance scale')
    parser.add_argument('--cg-scale',        type=float, default=0.5,
                        help='CG guidance scale')
    parser.add_argument('--output-dir',           type=str,   default='evaluation')
    parser.add_argument('--fid-extractor-path',   type=str,   default='checkpoints/fid_extractor.pth',
                        help='Path to MNISTFeatureExtractor checkpoint (train_fid_extractor.py)')
    parser.add_argument('--device',               type=str,   default='cuda')
    args = parser.parse_args()

    assert args.num_samples % 10 == 0, "--num-samples must be divisible by 10"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DIFFUSION MODEL EVALUATION")
    print("=" * 80)
    print(f"Guidance type : {args.guidance_type}")
    print(f"Num samples   : {args.num_samples}")
    print(f"Num steps     : {args.num_steps}")
    print(f"Device        : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    loader  = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    real_images, _ = next(iter(loader))
    real_images    = real_images[:args.num_samples]

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(num_timesteps=1000)
    scheduler.to(device)

    # ── Diffusion process visualization (always runs) ─────────────────────────
    print("\nVisualizing forward diffusion process...")
    sample_img, _ = dataset[0]
    visualize_diffusion_process(sample_img.unsqueeze(0), scheduler, args.output_dir, device)

    # ── Load classifier ────────────────────────────────────────────────────────
    if not os.path.exists(args.classifier_path):
        print(f"ERROR: Classifier checkpoint not found at {args.classifier_path}")
        print("  Run: python train/train_classifier.py  then retry.")
        return
    print(f"\nLoading classifier from {args.classifier_path}...")
    classifier = SimpleClassifier(num_classes=10)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier = classifier.to(device)

    # ── Load FID feature extractor (optional) ─────────────────────────────────
    fid_extractor = None
    if os.path.exists(args.fid_extractor_path):
        print(f"Loading FID extractor from {args.fid_extractor_path}...")
        fid_extractor = MNISTFeatureExtractor(feature_dim=256).to(device)
        fid_extractor.load_state_dict(torch.load(args.fid_extractor_path, map_location=device))
    else:
        print(f"FID extractor not found at {args.fid_extractor_path}.")
        print("  Run `python train_fid_extractor.py` to enable domain-specific FID.")

    # ── Load models ────────────────────────────────────────────────────────────
    cfg_model, cg_model = None, None

    if args.guidance_type in ('classifier-free', 'both'):
        print(f"Loading CFG model from {args.model_path}...")
        cfg_model = SimpleUNet(in_channels=1, out_channels=1, channels=64, num_classes=10)
        ckpt = torch.load(args.model_path, map_location=device)
        cfg_model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt)
        cfg_model = cfg_model.to(device)

    if args.guidance_type in ('classifier-guided', 'both'):
        print(f"Loading CG model from {args.cg_model_path}...")
        cg_model = SimpleUNet(in_channels=1, out_channels=1, channels=64, num_classes=10)
        ckpt = torch.load(args.cg_model_path, map_location=device)
        cg_model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt)
        cg_model = cg_model.to(device)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    results    = {}
    cfg_samples, cg_samples = None, None

    if args.guidance_type in ('classifier-free', 'both'):
        print("\n── CFG Evaluation ──")
        res = generate_and_evaluate(
            'classifier-free', cfg_model, classifier, scheduler,
            args.num_samples, args.num_steps, args.cfg_scale, device,
            eval_classifier=fid_extractor,
        )
        real_d = denormalize(real_images).clamp(0, 1)
        gen_d  = denormalize(res['samples']).clamp(0, 1)
        res['fid'] = compute_fid(real_d, gen_d)
        res['fid_domain'] = (
            compute_fid_domain(real_d, gen_d, fid_extractor, device)
            if fid_extractor is not None else None
        )
        cfg_samples       = res['samples']
        results['CFG']    = res

        save_per_class_grid(
            cfg_samples,
            title    = f'CFG Generated Samples (scale={args.cfg_scale})',
            filename = os.path.join(args.output_dir, 'cfg_per_class.png'),
            n_per_class = args.num_samples // 10,
        )

    if args.guidance_type in ('classifier-guided', 'both'):
        print("\n── CG Evaluation ──")
        res = generate_and_evaluate(
            'classifier-guided', cg_model, classifier, scheduler,
            args.num_samples, args.num_steps, args.cg_scale, device,
            cg_classifier=classifier,
            eval_classifier=fid_extractor,
        )
        real_d = denormalize(real_images).clamp(0, 1)
        gen_d  = denormalize(res['samples']).clamp(0, 1)
        res['fid'] = compute_fid(real_d, gen_d)
        res['fid_domain'] = (
            compute_fid_domain(real_d, gen_d, fid_extractor, device)
            if fid_extractor is not None else None
        )
        cg_samples     = res['samples']
        results['CG']  = res

        save_per_class_grid(
            cg_samples,
            title    = f'CG Generated Samples (scale={args.cg_scale})',
            filename = os.path.join(args.output_dir, 'cg_per_class.png'),
            n_per_class = args.num_samples // 10,
        )

    # ── Comparison grid ────────────────────────────────────────────────────────
    if cfg_samples is not None or cg_samples is not None:
        save_comparison_grid(real_images, cfg_samples, cg_samples, args.output_dir)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for method, res in results.items():
        print(f"\n  {method}:")
        print(f"    FID Score (pixel)  : {res['fid']:.2f}   (lower is better)")
        if res['fid_domain'] is not None:
            print(f"    FID Score (domain) : {res['fid_domain']:.2f}   (lower is better, feature-based)")
        print(f"    Inception Score    : {res['inception_score']:.4f}  (higher is better)")
        print(f"    Class Accuracy     : {res['class_accuracy']:.2f}%")

    if len(results) == 2:
        cfg_r, cg_r = results['CFG'], results['CG']
        print("\n  Comparison (CFG vs CG):")
        print(f"    Better FID (pixel)  : {'CFG' if cfg_r['fid'] < cg_r['fid'] else 'CG'}")
        if cfg_r['fid_domain'] is not None and cg_r['fid_domain'] is not None:
            print(f"    Better FID (domain) : {'CFG' if cfg_r['fid_domain'] < cg_r['fid_domain'] else 'CG'}")
        print(f"    Better IS           : {'CFG' if cfg_r['inception_score'] > cg_r['inception_score'] else 'CG'}")
        print(f"    Better Accuracy     : {'CFG' if cfg_r['class_accuracy'] > cg_r['class_accuracy'] else 'CG'}")

    print(f"\n  Note: Pixel FID uses raw pixels; domain FID uses MNIST feature extractor.")
    print(f"        Neither is comparable to published literature (different datasets).")
    print(f"\n  Output files saved to: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
