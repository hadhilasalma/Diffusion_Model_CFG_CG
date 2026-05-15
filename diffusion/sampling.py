"""
Sampling functions for CFG and CG diffusion models.

Imported by sample.py — keeps the CLI script thin and
separates generation logic from argument parsing.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def sample_cfg(
    model,
    scheduler,
    num_samples=100,
    class_labels=None,
    guidance_scale=7.5,
    device='cuda',
    num_steps=1000,
):
    """
    Generate samples using classifier-free guidance.

    Runs the U-Net twice per step (conditional + unconditional) and
    blends the outputs using the guidance scale.
    """
    model.eval()
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    null_labels = torch.full((num_samples,), 10, dtype=torch.long, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CFG)"):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        noise_uncond = model(x_t, t_batch, null_labels)

        if class_labels is not None and guidance_scale != 1.0:
            noise_cond = model(x_t, t_batch, class_labels)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = noise_uncond

        alpha_t     = scheduler.alphas[t]
        alpha_bar_t = scheduler.alphas_cumprod[t]
        x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

        if t > 0:
            # σ_t = √β̃_t (posterior variance) — DDPM paper Algorithm 2
            x_t = x_t + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x_t)

    return x_t


def sample_cg(
    model,
    classifier,
    scheduler,
    num_samples=100,
    class_labels=None,
    guidance_scale=0.5,
    device='cuda',
    num_steps=1000,
):
    """
    Generate samples using classifier guidance.

    Uses the unconditional U-Net for denoising, then steers each step
    using the gradient of a noise-aware classifier.
    """
    model.eval()
    classifier.eval()

    if class_labels is None:
        raise ValueError("class_labels required for classifier-guided sampling")

    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CG)"):
        t_batch    = torch.full((num_samples,), t, dtype=torch.long, device=device)
        null_labels = torch.full((num_samples,), 10, dtype=torch.long, device=device)

        with torch.no_grad():
            noise_pred = model(x_t, t_batch, null_labels)

        # Classifier gradient
        x_t_grad   = x_t.clone().requires_grad_(True)
        log_probs  = F.log_softmax(classifier(x_t_grad, t_batch), dim=1)
        class_prob = log_probs[range(num_samples), class_labels].sum()
        grad       = torch.autograd.grad(class_prob, x_t_grad)[0]

        grad = torch.clamp(grad, -10.0, 10.0)

        with torch.no_grad():
            alpha_t     = scheduler.alphas[t]
            alpha_bar_t = scheduler.alphas_cumprod[t]
            # √(1−ᾱ_t) converts classifier gradient from score space to noise space
            # ε̂ = ε_θ − √(1−ᾱ_t) × s × ∇log p(y|x_t)  — Dhariwal & Nichol 2021
            noise_pred  = noise_pred - guidance_scale * torch.sqrt(1 - alpha_bar_t) * grad
            x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

            if t > 0:
                # σ_t = √β̃_t (posterior variance) — DDPM paper Algorithm 2
                x_t = x_t + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x_t)

    return x_t
