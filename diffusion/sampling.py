"""
Sampling (Inference)
====================
Two functions — one per guidance method.

Both start from pure Gaussian noise x_T ~ N(0, I) and iteratively denoise
over T steps using the DDPM reverse formula:

    x_{t-1} = ( x_t − (1−α_t)/√(1−ᾱ_t) · ε̂ ) / √α_t  +  √β̃_t · z

The only difference is HOW ε̂ (the noise prediction) is computed each step.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── Classifier-Free Guidance ───────────────────────────────────────────────────

@torch.no_grad()
def sample_cfg(model, scheduler, num_samples=100,
               class_labels=None, guidance_scale=7.5,
               device='cuda', num_steps=1000):
    """
    Generate samples using Classifier-Free Guidance.

    The CFG model runs TWICE per step — conditional (target label) and
    unconditional (null token label=10). Outputs are blended:

        ε̂ = ε_uncond + s · (ε_cond − ε_uncond)

    Higher s → stronger class signal, less diversity.
    Recommended s ≈ 7.5 for MNIST.

    Args:
        model:          CFG-trained SimpleUNet
        scheduler:      DDPMScheduler
        num_samples:    Number of images to generate
        class_labels:   Target class [num_samples], or None for unconditional
        guidance_scale: Blending strength s
        device:         'cuda' or 'cpu'
        num_steps:      Denoising steps (1000 = full quality, 200 = fast)
    Returns:
        x: Generated images [num_samples, 1, 28, 28] in range [−1, 1]
    """
    model.eval()
    x    = torch.randn(num_samples, 1, 28, 28, device=device)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CFG)"):
        tb = torch.full((num_samples,), t, dtype=torch.long, device=device)

        e_uncond = model(x, tb, null)

        if class_labels is not None and guidance_scale != 1.0:
            e_cond = model(x, tb, class_labels)
            e_pred = e_uncond + guidance_scale * (e_cond - e_uncond)
        else:
            e_pred = e_uncond

        # DDPM reverse step
        a, ab = scheduler.alphas[t], scheduler.alphas_cumprod[t]
        x = (x - (1 - a) / torch.sqrt(1 - ab) * e_pred) / torch.sqrt(a)
        if t > 0:
            # Add back small noise (stochastic — keeps samples diverse)
            x = x + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x)

    return x


# ── Classifier Guidance ────────────────────────────────────────────────────────

def sample_cg(model, classifier, scheduler, num_samples=100,
              class_labels=None, guidance_scale=0.5,
              device='cuda', num_steps=1000):
    """
    Generate samples using Classifier Guidance (Dhariwal & Nichol 2021).

    Uses an unconditional diffusion model for denoising, then steers each
    step with the gradient of a noise-aware classifier:

        ε̂ = ε_θ(x_t) − √(1−ᾱ_t) · s · ∇log p(y | x_t)

    The gradient points toward pixels that increase the target class
    probability. √(1−ᾱ_t) converts it from pixel space to noise space.
    Gradient is clamped to [−10, 10] to prevent instability at high noise.

    Recommended s ≈ 0.5 for MNIST.

    Args:
        model:          Unconditional SimpleUNet (trained with mode='cg')
        classifier:     Noise-aware SimpleClassifier
        scheduler:      DDPMScheduler
        num_samples:    Number of images to generate
        class_labels:   Target class [num_samples] — required
        guidance_scale: Gradient strength s
        device:         'cuda' or 'cpu'
        num_steps:      Denoising steps
    Returns:
        x: Generated images [num_samples, 1, 28, 28] in range [−1, 1]
    """
    if class_labels is None:
        raise ValueError("class_labels is required for classifier guidance")

    model.eval()
    classifier.eval()
    x    = torch.randn(num_samples, 1, 28, 28, device=device)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CG)"):
        tb = torch.full((num_samples,), t, dtype=torch.long, device=device)

        # 1. Unconditional noise prediction
        with torch.no_grad():
            e_pred = model(x, tb, null)

        # 2. Classifier gradient — clone so grad doesn't affect x
        x_grad    = x.clone().requires_grad_(True)
        log_probs = F.log_softmax(classifier(x_grad, tb), dim=1)
        log_probs[range(num_samples), class_labels].sum().backward()
        grad = torch.clamp(x_grad.grad, -10.0, 10.0)

        # 3. Steer noise prediction, then DDPM reverse step
        with torch.no_grad():
            a, ab  = scheduler.alphas[t], scheduler.alphas_cumprod[t]
            e_pred = e_pred - guidance_scale * torch.sqrt(1 - ab) * grad
            x      = (x - (1 - a) / torch.sqrt(1 - ab) * e_pred) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x)

    return x
