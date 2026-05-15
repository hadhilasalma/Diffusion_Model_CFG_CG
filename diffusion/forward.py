"""
Forward and Reverse Process Utilities

Provides functions for the forward (noising) and reverse (denoising) processes
in diffusion models.
"""

import torch
import torch.nn.functional as F


def forward_process(x0, t, scheduler, noise):
    """
    Forward diffusion process: Add noise to clean image.

    Args:
        x0: Clean image [batch_size, channels, height, width]
        t: Timesteps [batch_size]
        scheduler: Noise scheduler (DDPMScheduler)
        noise: Random Gaussian noise [batch_size, channels, height, width]

    Returns:
        x_t: Noisy image at timestep t
    """
    return scheduler.forward_process(x0, t, noise)


@torch.no_grad()
def reverse_process_ddpm(model, scheduler, num_steps=1000, batch_size=1,
                        device='cuda', class_label=None, guidance_scale=7.5):
    """
    Reverse diffusion process (DDPM): Generate image from noise.

    Args:
        model: Trained diffusion model
        scheduler: Noise scheduler
        num_steps: Number of denoising steps
        batch_size: Number of images to generate
        device: Device to use
        class_label: Class label for conditional generation [batch_size]
                    or None for unconditional
        guidance_scale: Classifier-free guidance scale (7.5 recommended)

    Returns:
        x_0: Generated images [batch_size, channels, height, width]
    """
    # Start from pure noise
    x_t = torch.randn(batch_size, 1, 28, 28, device=device)

    # Denoise step by step
    for t in range(num_steps - 1, -1, -1):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Get noise prediction
        if class_label is not None:
            # Classifier-free guidance
            with torch.no_grad():
                # Conditional prediction
                noise_cond = model(x_t, t_batch, class_label)

                # Unconditional prediction
                null_class = torch.full_like(class_label, 10)  # 10 = unconditional token
                noise_uncond = model(x_t, t_batch, null_class)

                # Blend with guidance
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            # Unconditional generation
            with torch.no_grad():
                null_class = torch.full((batch_size,), 10, dtype=torch.long, device=device)
                noise_pred = model(x_t, t_batch, null_class)

        # Denoise step
        alpha_t = scheduler.alphas[t]
        alpha_bar_t = scheduler.alphas_cumprod[t]

        # x_t = (x_t - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_t)
        x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

        # Add small noise (except on last step)
        if t > 0:
            z = torch.randn_like(x_t)
            beta_t = scheduler.betas[t]
            x_t = x_t + torch.sqrt(beta_t) * z

    return x_t


@torch.no_grad()
def reverse_process_ddim(model, scheduler, num_steps=100, batch_size=1,
                        device='cuda', class_label=None, guidance_scale=7.5):
    """
    DDIM sampling: Faster reverse process (skip timesteps).

    DDIM can generate good samples in 10-50 steps instead of 1000.

    Args:
        model: Trained diffusion model
        scheduler: Noise scheduler
        num_steps: Number of denoising steps (e.g., 50 instead of 1000)
        batch_size: Number of images
        device: Device to use
        class_label: Class label or None
        guidance_scale: Classifier-free guidance scale

    Returns:
        x_0: Generated images
    """
    # Start from noise
    x_t = torch.randn(batch_size, 1, 28, 28, device=device)

    # Sample timesteps to use (skip many)
    timesteps = torch.linspace(0, 1000 - 1, num_steps, device=device).long()

    # Reverse order (from T to 0)
    timesteps = timesteps.flip(0)

    for i, t in enumerate(timesteps):
        t = t.item()
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Get noise prediction
        if class_label is not None:
            with torch.no_grad():
                noise_cond = model(x_t, t_batch, class_label)
                null_class = torch.full_like(class_label, 10)
                noise_uncond = model(x_t, t_batch, null_class)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            with torch.no_grad():
                noise_pred = model(x_t, t_batch, None)

        # DDIM step (deterministic)
        alpha_bar_t = scheduler.alphas_cumprod[t]
        x_t = (x_t - noise_pred * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()

    return x_t


@torch.no_grad()
def reverse_process_classifier_guided(model, classifier, scheduler, num_steps=1000,
                                     batch_size=1, device='cuda', class_label=None,
                                     guidance_scale=0.5):
    """
    Reverse process with classifier-guided diffusion.

    Uses a separate classifier to guide the denoising process.

    Args:
        model: Trained diffusion model
        classifier: Trained classifier for guidance
        scheduler: Noise scheduler
        num_steps: Number of steps
        batch_size: Number of images
        device: Device to use
        class_label: Target class [batch_size]
        guidance_scale: Guidance scale

    Returns:
        x_0: Generated images
    """
    if class_label is None:
        raise ValueError("class_label required for classifier-guided diffusion")

    # Start from noise
    x_t = torch.randn(batch_size, 1, 28, 28, device=device)

    for t in range(num_steps - 1, -1, -1):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Get noise prediction from diffusion model (unconditional)
        with torch.no_grad():
            null_class = torch.full((batch_size,), 10, dtype=torch.long, device=device)
            noise_pred = model(x_t, t_batch, null_class)

        # Get classifier guidance
        x_t_grad = x_t.clone().requires_grad_(True)
        logits = classifier(x_t_grad)
        log_probs = F.log_softmax(logits, dim=1)
        class_prob = log_probs[range(batch_size), class_label].sum()

        # Compute gradient
        grad = torch.autograd.grad(class_prob, x_t_grad)[0]

        with torch.no_grad():
            # Adjust noise prediction by gradient
            noise_pred_adjusted = noise_pred - guidance_scale * grad

            # Denoise step
            alpha_t = scheduler.alphas[t]
            alpha_bar_t = scheduler.alphas_cumprod[t]

            x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred_adjusted) / torch.sqrt(alpha_t)

            # Add noise
            if t > 0:
                z = torch.randn_like(x_t)
                beta_t = scheduler.betas[t]
                x_t = x_t + torch.sqrt(beta_t) * z

    return x_t
