"""
DDPM Noise Scheduler
====================
Implements the linear beta schedule from Ho et al. 2020.

Key idea: instead of looping t times to add noise, the reparameterisation
trick lets you jump directly from x0 to any noisy x_t in one shot:

    x_t = √ᾱ_t · x₀  +  √(1 − ᾱ_t) · ε      ε ~ N(0, I)

The scheduler precomputes all noise-level constants at init time.
"""

import torch


class DDPMScheduler:
    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end:   float = 0.02):
        self.num_timesteps = num_timesteps

        # β_t: how much noise is added at each step (linearly increasing)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # α_t = 1 − β_t: fraction of signal that survives each step
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏α_i: cumulative signal fraction from step 0 to t
        # ᾱ_0 ≈ 1 (clean image),  ᾱ_T ≈ 0 (pure noise)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # √ᾱ_t and √(1−ᾱ_t) — used directly in forward_process
        self.sqrt_alphas_cumprod           = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # ᾱ_{t-1} shifted by one step, needed for posterior variance
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]], dim=0
        )

        # β̃_t: variance of noise re-added during each reverse step
        # Keeps sampling stochastic so each run gives a different image
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)

    # ── Device transfer ────────────────────────────────────────────────────────

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "alphas_cumprod_prev", "posterior_variance",
            "posterior_log_variance_clipped",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ── Forward process (add noise) ────────────────────────────────────────────

    def forward_process(self, x0: torch.Tensor,
                        t:   torch.Tensor,
                        noise: torch.Tensor) -> torch.Tensor:
        """
        Jump directly from clean image x0 to noisy x_t at timestep t.

        Formula:  x_t = √ᾱ_t · x₀  +  √(1−ᾱ_t) · ε

        Args:
            x0:    Clean image  [B, C, H, W], values in [−1, 1]
            t:     Timestep     [B]
            noise: Gaussian ε   [B, C, H, W]
        Returns:
            x_t:   Noisy image  [B, C, H, W]
        """
        B = x0.shape[0]
        sqrt_ab  = self.sqrt_alphas_cumprod[t].reshape(B, 1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1ab * noise
