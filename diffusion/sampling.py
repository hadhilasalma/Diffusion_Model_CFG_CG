"""
Sampling (Inference) — CFG and Classifier Guidance

Shared reverse step:
    x_{t-1} = (x_t − (1−α_t)/√(1−ᾱ_t) · ε̂) / √α_t  +  √β̃_t · z

The only difference between CFG and CG is how ε̂ is computed each step.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _ddpm_step(x, e_pred, t, scheduler):
    """One DDPM reverse step: denoise a little, then add a little noise back."""
    a  = scheduler.alphas[t]
    ab = scheduler.alphas_cumprod[t]
    x  = (x - (1 - a) / torch.sqrt(1 - ab) * e_pred) / torch.sqrt(a)
    if t > 0:
        x = x + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x)
    return x


# ── Classifier-Free Guidance ───────────────────────────────────────────────────

@torch.no_grad()
def sample_cfg(model, scheduler, num_samples=100,
               class_labels=None, guidance_scale=7.5,
               device='cuda', num_steps=1000):
    """
    Classifier-Free Guidance sampling.

    The model runs twice per step — unconditional (null token) and conditional
    (target label) — then blends the two predictions:

        ε̂ = ε_uncond + s · (ε_cond − ε_uncond)

    Higher s → stronger class signal, less diversity. Recommended s ≈ 7.5.
    Returns generated images [num_samples, 1, 28, 28] in range [−1, 1].
    """
    model.eval()
    x    = torch.randn(num_samples, 1, 28, 28, device=device)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CFG)"):
        tb       = torch.full((num_samples,), t, dtype=torch.long, device=device)
        e_uncond = model(x, tb, null)

        if class_labels is not None and guidance_scale != 1.0:
            e_cond = model(x, tb, class_labels)
            e_pred = e_uncond + guidance_scale * (e_cond - e_uncond)
        else:
            e_pred = e_uncond

        x = _ddpm_step(x, e_pred, t, scheduler)

    return x


# ── Classifier Guidance ────────────────────────────────────────────────────────

def sample_cg(model, classifier, scheduler, num_samples=100,
              class_labels=None, guidance_scale=0.5,
              device='cuda', num_steps=1000):
    """
    Classifier Guidance sampling (Dhariwal & Nichol 2021).

    Uses an unconditional denoiser, then steers each step using the gradient
    of a noise-aware classifier:

        ε̂ = ε_θ(x_t) − √(1−ᾱ_t) · s · ∇log p(y | x_t)

    Gradient is clamped to [−10, 10] to prevent instability at high noise.
    Recommended s ≈ 0.5. Requires class_labels — raises ValueError if None.
    Returns generated images [num_samples, 1, 28, 28] in range [−1, 1].
    """
    if class_labels is None:
        raise ValueError("class_labels is required for classifier guidance")

    model.eval()
    classifier.eval()
    x    = torch.randn(num_samples, 1, 28, 28, device=device)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=device)

    for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling (CG)"):
        tb = torch.full((num_samples,), t, dtype=torch.long, device=device)

        # Step 1 — unconditional noise prediction (no gradient needed)
        with torch.no_grad():
            e_pred = model(x, tb, null)

        # Step 2 — gradient of log p(y | x_t) w.r.t. x_t
        x_grad = x.clone().requires_grad_(True)
        F.log_softmax(classifier(x_grad, tb), dim=1)[range(num_samples), class_labels].sum().backward()
        grad = torch.clamp(x_grad.grad, -10.0, 10.0)

        # Step 3 — steer the noise prediction, then take the reverse step
        with torch.no_grad():
            ab     = scheduler.alphas_cumprod[t]
            e_pred = e_pred - guidance_scale * torch.sqrt(1 - ab) * grad
            x      = _ddpm_step(x, e_pred, t, scheduler)

    return x
