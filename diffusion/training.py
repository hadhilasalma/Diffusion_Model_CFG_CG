"""
Diffusion Training Loop
=======================
Single function train_diffusion() shared by both training scripts.

The only difference between CFG and CG training is the label strategy:
    mode='cfg'  →  10% label dropout (null token)   — trains CFG model
    mode='cg'   →  always null token                 — trains unconditional model
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def train_diffusion(model, train_loader, scheduler,
                    mode='cfg', num_epochs=5, learning_rate=1e-4,
                    device='cuda', checkpoint_dir='checkpoints',
                    val_loader=None):
    """
    Train a diffusion U-Net.

    Args:
        model:          SimpleUNet instance
        train_loader:   DataLoader for MNIST training split
        scheduler:      DDPMScheduler
        mode:           'cfg'  — 10% label dropout (CFG training)
                        'cg'   — always null token  (unconditional for CG)
        num_epochs:     Training epochs
        learning_rate:  Adam lr (default 1e-4)
        device:         'cuda' or 'cpu'
        checkpoint_dir: Directory to save .pth files
        val_loader:     Optional validation DataLoader
    Returns:
        model, optimizer, train_losses, val_losses
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_sched  = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scheduler.to(device)

    # Mixed precision — runs forward in fp16, keeps gradients in fp32
    use_amp = str(device) != 'cpu'
    scaler  = GradScaler(enabled=use_amp)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            B = images.shape[0]

            # Sample a random timestep and noise for each image
            t     = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(images)
            x_t   = scheduler.forward_process(images, t, noise)

            # Label strategy: CFG uses dropout, CG always uses null token
            if mode == 'cfg':
                labels_in = labels.clone()
                labels_in[torch.rand(B, device=device) < 0.1] = 10   # 10% dropout
            else:
                labels_in = torch.full((B,), 10, dtype=torch.long, device=device)

            # Forward + loss
            with autocast(enabled=use_amp):
                noise_pred = model(x_t, t, labels_in)
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        lr_sched.step()
        print(f"\nEpoch {epoch+1}  train_loss={avg_train:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}", end="")

        # ── Validation ────────────────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    B = images.shape[0]
                    t     = torch.randint(0, 1000, (B,), device=device)
                    noise = torch.randn_like(images)
                    x_t   = scheduler.forward_process(images, t, noise)
                    null  = torch.full((B,), 10, dtype=torch.long, device=device)
                    val_loss += F.mse_loss(model(x_t, t, null), noise).item()

            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)
            print(f"  val_loss={avg_val:.4f}", end="")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': train_losses,
                    'val_loss_history': val_losses,
                }, os.path.join(checkpoint_dir, 'model_best.pth'))
                print("  [best saved]", end="")

            model.train()

        print()

        # ── Per-epoch checkpoint ───────────────────────────────────────────────
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': train_losses,
            'val_loss_history': val_losses if val_loader else None,
            'scheduler': {
                'betas':           scheduler.betas,
                'alphas':          scheduler.alphas,
                'alphas_cumprod':  scheduler.alphas_cumprod,
                'num_timesteps':   scheduler.num_timesteps,
            },
        }, os.path.join(checkpoint_dir, 'model_checkpoint.pth'))

    # Load best weights back if validation was used
    best_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if val_loader is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nRestored best model from epoch {ckpt['epoch']}")

    return model, optimizer, train_losses, val_losses
