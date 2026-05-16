"""
DDPM Noise Scheduler

Controls how much noise to add at each timestep during the diffusion process.
Provides both forward process (add noise) and reverse process (remove noise) utilities.

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch


class DDPMScheduler:
    """
    DDPM noise schedule.

    Controls the noise schedule for the forward (diffusion) process.
    The forward process gradually adds Gaussian noise to an image until
    it becomes pure noise.

    Key insight: We don't train the forward process - it's deterministic!
    We only train the reverse process (the model).
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initialize DDPM scheduler with linear noise schedule.

        Args:
            num_timesteps: Number of diffusion steps (typically 1000)
                          More steps = better quality but slower
            beta_start: Starting variance (small, almost no noise)
            beta_end: Ending variance (large, almost all noise)

        The schedule is LINEAR: β_t = beta_start + (beta_end - beta_start) * t/T
        """
        self.num_timesteps = num_timesteps

        # Create linear schedule from beta_start to beta_end
        # β_t = variance added at step t
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # α_t = 1 - β_t (how much of original image remains)
        self.alphas = 1.0 - self.betas

        # ᾱ_t = cumulative product of alphas
        # This is key! It tells us the final noise level after t steps
        # ᾱ_t = α_0 * α_1 * ... * α_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # sqrt versions (used in forward process formula)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For reverse process (denoising)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]], dim=0
        )

        # β̃_t = variance for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)


    def to(self, device):
        """Move all tensors to specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def forward_process(self, x0, t, noise):
        """
        Add noise to image x0 at timestep t (forward diffusion process).

        This is the **deterministic** forward process that we don't train!
        Given x0, we can directly compute x_t for any t without iterating.

        Formula: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        Args:
            x0: Clean image [batch_size, channels, height, width]
               Values should be in range [-1, 1]
            t: Timestep indices [batch_size]
               Each image can be at a different timestep
            noise: Random Gaussian noise [batch_size, channels, height, width]
                  Sampled from N(0, I)

        Returns:
            x_t: Noisy image at timestep t [batch_size, channels, height, width]
        """
        # Validate inputs
        assert x0.dim() == 4, f"Expected x0 to have 4 dimensions, got {x0.dim()}"
        assert t.dim() == 1, f"Expected t to have 1 dimension, got {t.dim()}"
        assert noise.shape == x0.shape, f"Noise shape {noise.shape} must match x0 shape {x0.shape}"
        assert x0.shape[0] == t.shape[0], f"Batch size mismatch: x0 has {x0.shape[0]}, t has {t.shape[0]}"
        assert t.min() >= 0 and t.max() < self.num_timesteps, f"Timesteps must be in range [0, {self.num_timesteps})"

        batch_size = x0.shape[0]

        # Get the sqrt_alpha_cumprod for this timestep
        # Reshape to [batch_size, 1, 1, 1] for broadcasting with images
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)

        # Apply formula: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        # This mixes the original image with noise
        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t



if __name__ == "__main__":
    # Test the scheduler
    scheduler = DDPMScheduler(num_timesteps=1000)
    scheduler.to('cpu')

    # Create dummy input
    x0 = torch.randn(2, 1, 28, 28)  # Clean image
    t = torch.tensor([100, 500])     # Timesteps
    noise = torch.randn_like(x0)     # Random noise

    # Forward process
    x_t = scheduler.forward_process(x0, t, noise)

    print(f"x0 range: [{x0.min():.2f}, {x0.max():.2f}]")
    print(f"x_t range: [{x_t.min():.2f}, {x_t.max():.2f}]")
    print(f"√(ᾱ_t) at t=100: {scheduler.sqrt_alphas_cumprod[100]:.4f}")
    print(f"√(1-ᾱ_t) at t=100: {scheduler.sqrt_one_minus_alphas_cumprod[100]:.4f}")
    print(f"√(ᾱ_t) at t=500: {scheduler.sqrt_alphas_cumprod[500]:.4f}")
    print(f"√(1-ᾱ_t) at t=500: {scheduler.sqrt_one_minus_alphas_cumprod[500]:.4f}")
