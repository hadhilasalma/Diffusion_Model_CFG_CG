"""
Shared diffusion training loop.

Both train.py (CFG) and train_cg.py (unconditional) call train_diffusion()
from here — the only difference between the two modes is label handling,
controlled by the `mode` argument.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def train_diffusion(
    model,
    train_loader,
    scheduler,
    mode='cfg',
    num_epochs=5,
    learning_rate=1e-4,
    device='cuda',
    checkpoint_dir='checkpoints',
    val_loader=None,
):
    """
    Train a diffusion U-Net.

    Args:
        mode: 'cfg' — 10% label dropout (classifier-free guidance training)
              'cg'  — always null token 10 (pure unconditional training for CG)
        val_loader: Optional validation dataloader
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    use_amp = str(device) != 'cpu'
    scaler = GradScaler(enabled=use_amp)

    scheduler.to(device)
    model = model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            t = torch.randint(0, 1000, (batch_size,), device=device)
            noise = torch.randn_like(images)
            x_t = scheduler.forward_process(images, t, noise)

            if mode == 'cfg':
                # 10% label dropout — model learns both conditional and unconditional
                mask = torch.rand(batch_size, device=device) < 0.1
                labels_in = labels.clone()
                labels_in[mask] = 10
            else:
                # Always null token — pure unconditional model for CG
                labels_in = torch.full((batch_size,), 10, dtype=torch.long, device=device)

            with autocast(enabled=use_amp):
                noise_pred = model(x_t, t, labels_in)
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - LR: {current_lr:.2e}", end="")

        # --- VALIDATION ---
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    batch_size = images.shape[0]

                    t = torch.randint(0, 1000, (batch_size,), device=device)
                    noise = torch.randn_like(images)
                    x_t = scheduler.forward_process(images, t, noise)

                    labels_in = torch.full((batch_size,), 10, dtype=torch.long, device=device)

                    noise_pred = model(x_t, t, labels_in)
                    val_loss = F.mse_loss(noise_pred, noise)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f" | Val Loss: {avg_val_loss:.4f}", end="")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(checkpoint_dir, 'model_best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': train_losses,
                    'val_loss_history': val_losses,
                }, best_path)
                print(" [Best model saved]", end="")

            model.train()

        print()  # Newline

        # --- CHECKPOINT ---
        checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': train_losses,
            'val_loss_history': val_losses if val_loader else None,
            'scheduler': {
                'betas': scheduler.betas,
                'alphas': scheduler.alphas,
                'alphas_cumprod': scheduler.alphas_cumprod,
                'num_timesteps': scheduler.num_timesteps,
            },
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    if val_loader is not None and os.path.exists(os.path.join(checkpoint_dir, 'model_best.pth')):
        best_ckpt = torch.load(os.path.join(checkpoint_dir, 'model_best.pth'))
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"\nLoaded best model from epoch {best_ckpt['epoch']}")

    return model, optimizer, train_losses, val_losses
