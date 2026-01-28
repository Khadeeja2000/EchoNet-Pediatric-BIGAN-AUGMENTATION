import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import cv2


class EchoDataset(Dataset):
    def __init__(self, manifest_csv: str, frames: int = 32, size: int = 64):
        self.df = pd.read_csv(manifest_csv)
        if "processed_path" in self.df.columns:
            self.df = self.df[self.df["processed_path"].astype(str).str.len() > 0].reset_index(drop=True)
        self.frames = frames
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["processed_path"] if "processed_path" in row else row["file_path"]
        sex_val = str(row["sex"]).strip().lower()
        sex = 0 if sex_val.startswith("f") else 1
        age_bin = row["age_bin"] if "age_bin" in row else "other"

        age_mapping = {"0-1": 0, "2-5": 1, "6-10": 2, "11-15": 3, "16-18": 4, "other": 5}
        cond = torch.tensor([sex, age_mapping.get(str(age_bin), 5)], dtype=torch.float32)

        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total is None or total <= 0:
            total = self.frames
        indices = np.linspace(0, max(total - 1, 0), self.frames).astype(int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            f = cv2.resize(f, (self.size, self.size))
            frames.append(f)
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((self.size, self.size), dtype=np.uint8) for _ in range(self.frames)]
        if len(frames) < self.frames:
            frames += [frames[-1]] * (self.frames - len(frames))

        arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]
        x = torch.from_numpy(arr).unsqueeze(0)
        return x, cond


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1, size: int = 64):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)

        layers = []
        # 4 -> 8 -> 16 -> 32
        layers.extend([
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
        ])
        
        if size >= 64:
            # 32 -> 64 (spatial-only if needed)
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.ConvTranspose3d(32, channels, 3, 1, 1),
            ])
        else:
            layers.append(nn.ConvTranspose3d(64, channels, 3, 1, 1))
        
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4, 4)
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1, size: int = 64):
        super().__init__()
        layers = []
        if size >= 64:
            # 64 -> 32 (spatial-only)
            layers.extend([
                nn.Conv3d(channels, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        else:
            layers.extend([
                nn.Conv3d(channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
        # 32 -> 16 -> 8 -> 4
        layers.extend([
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(512 * 4 * 4 * 4 + cond_dim, z_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1, size: int = 64, use_sn: bool = True):
        super().__init__()
        self.use_sn = use_sn
        
        conv_layers = []
        if size >= 64:
            # 64 -> 32 (spatial-only)
            conv = nn.Conv3d(channels, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
            if use_sn:
                conv = spectral_norm(conv)
            conv_layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
        else:
            conv = nn.Conv3d(channels, 64, 4, 2, 1)
            if use_sn:
                conv = spectral_norm(conv)
            conv_layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
        
        # 32 -> 16
        conv = nn.Conv3d(64, 128, 4, 2, 1)
        if use_sn:
            conv = spectral_norm(conv)
        conv_layers.extend([
            conv,
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # 16 -> 8
        conv = nn.Conv3d(128, 256, 4, 2, 1)
        if use_sn:
            conv = spectral_norm(conv)
        conv_layers.extend([
            conv,
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # 8 -> 4
        conv = nn.Conv3d(256, 512, 4, 2, 1)
        if use_sn:
            conv = spectral_norm(conv)
        conv_layers.extend([
            conv,
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        self.video_path = nn.Sequential(*conv_layers)
        
        # FC layers with spectral norm
        fc1 = nn.Linear(512 * 4 * 4 * 4 + z_dim + cond_dim, 512)
        fc2 = nn.Linear(512, 1)
        if use_sn:
            fc1 = spectral_norm(fc1)
            fc2 = spectral_norm(fc2)
        
        self.fc = nn.Sequential(
            fc1,
            nn.LeakyReLU(0.2, inplace=True),
            fc2,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.video_path(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, z, c], dim=1)
        return self.fc(x)


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train(cfg: dict) -> None:
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Dataset
    dataset = EchoDataset(cfg["manifest"], frames=cfg["frames"], size=cfg["size"])
    loader = DataLoader(
        dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        num_workers=cfg["num_workers"], 
        drop_last=True,
        pin_memory=(device == "cuda")
    )
    print(f"Dataset loaded: {len(dataset)} samples, {len(loader)} batches per epoch")

    # Models
    G = Generator(cfg["z_dim"], cfg["cond_dim"], size=cfg["size"]).to(device)
    E = Encoder(cfg["z_dim"], cfg["cond_dim"], size=cfg["size"]).to(device)
    D = Discriminator(cfg["z_dim"], cfg["cond_dim"], size=cfg["size"], use_sn=cfg["use_sn"]).to(device)
    
    # EMA for generator
    ema_G = EMA(G, decay=cfg["ema_decay"]) if cfg["use_ema"] else None
    
    # Optimizers with different learning rates
    opt_G = optim.Adam(
        list(G.parameters()) + list(E.parameters()), 
        lr=cfg["lr_ge"], 
        betas=(cfg["beta1"], 0.999)
    )
    opt_D = optim.Adam(
        D.parameters(), 
        lr=cfg["lr_d"], 
        betas=(cfg["beta1"], 0.999)
    )
    
    # Learning rate schedulers
    if cfg["lr_decay_epoch"] > 0:
        sched_G = optim.lr_scheduler.StepLR(opt_G, step_size=cfg["lr_decay_epoch"], gamma=cfg["lr_gamma"])
        sched_D = optim.lr_scheduler.StepLR(opt_D, step_size=cfg["lr_decay_epoch"], gamma=cfg["lr_gamma"])
    else:
        sched_G = None
        sched_D = None

    # Hinge loss functions
    def d_hinge_loss(real_logits, fake_logits, label_smoothing=0.0):
        """Discriminator hinge loss with optional label smoothing"""
        target_real = 1.0 - label_smoothing
        loss_real = torch.relu(target_real - real_logits).mean()
        loss_fake = torch.relu(1.0 + fake_logits).mean()
        return loss_real + loss_fake

    def ge_hinge_loss(fake_logits, real_logits):
        """Generator/Encoder hinge loss"""
        return (-fake_logits).mean() + (-real_logits).mean()

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  LR (G/E): {cfg['lr_ge']}, LR (D): {cfg['lr_d']}")
    print(f"  Spectral Norm: {cfg['use_sn']}")
    print(f"  Label Smoothing: {cfg['label_smoothing']}")
    print(f"  EMA: {cfg['use_ema']} (decay={cfg['ema_decay']})")
    print(f"  Gradient Clipping: {cfg['clip_grad']}")
    print(f"  n_critic: {cfg['n_critic']}")
    print()

    # Training loop
    last_loss_GE = 0.0
    best_d_loss = float('inf')
    divergence_counter = 0
    
    for epoch in range(cfg["epochs"]):
        G.train()
        E.train()
        D.train()
        
        epoch_d_losses = []
        epoch_ge_losses = []
        
        for i, (real_x, cond) in enumerate(loader):
            real_x = real_x.to(device)
            cond = cond.to(device)
            batch_size = real_x.size(0)
            
            # Sample random latent vectors
            z = torch.randn(batch_size, cfg["z_dim"], device=device)
            
            # ============ Train Discriminator ============
            opt_D.zero_grad()
            
            # Encode real data
            with torch.no_grad():
                enc_z = E(real_x, cond)
            real_logits = D(real_x, enc_z, cond)
            
            # Generate fake data
            with torch.no_grad():
                fake_x = G(z, cond)
            fake_logits = D(fake_x, z, cond)
            
            # Discriminator loss with label smoothing
            d_loss = d_hinge_loss(real_logits, fake_logits, label_smoothing=cfg["label_smoothing"])
            
            # Check for NaN/Inf
            if torch.isnan(d_loss) or torch.isinf(d_loss):
                print(f"WARNING: D loss is {d_loss.item()}, skipping batch")
                continue
            
            d_loss.backward()
            
            # Gradient clipping
            if cfg["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg["clip_grad"])
            
            opt_D.step()
            epoch_d_losses.append(d_loss.item())
            
            # ============ Train Generator & Encoder ============
            if i % cfg["n_critic"] == 0:
                opt_G.zero_grad()
                
                # Generate fake samples
                fake_x = G(z, cond)
                fake_logits = D(fake_x, z, cond)
                
                # Encode real samples
                enc_z = E(real_x, cond)
                real_logits = D(real_x, enc_z, cond)
                
                # Generator/Encoder loss
                ge_loss = ge_hinge_loss(fake_logits, real_logits)
                
                # Check for NaN/Inf
                if torch.isnan(ge_loss) or torch.isinf(ge_loss):
                    print(f"WARNING: GE loss is {ge_loss.item()}, skipping batch")
                    continue
                
                last_loss_GE = ge_loss.item()
                ge_loss.backward()
                
                # Gradient clipping
                if cfg["clip_grad"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(G.parameters()) + list(E.parameters()), 
                        cfg["clip_grad"]
                    )
                
                opt_G.step()
                
                # Update EMA
                if ema_G is not None:
                    ema_G.update()
                
                epoch_ge_losses.append(ge_loss.item())
            
            # Logging
            if i % cfg["log_interval"] == 0:
                print(f"[Epoch {epoch}/{cfg['epochs']}] [Batch {i}/{len(loader)}] "
                      f"D_loss: {d_loss.item():.4f}, GE_loss: {last_loss_GE:.4f}")
        
        # Epoch summary
        avg_d_loss = np.mean(epoch_d_losses) if epoch_d_losses else 0.0
        avg_ge_loss = np.mean(epoch_ge_losses) if epoch_ge_losses else 0.0
        print(f"\n[Epoch {epoch} Summary] Avg D_loss: {avg_d_loss:.4f}, Avg GE_loss: {avg_ge_loss:.4f}")
        
        # Check for divergence
        if avg_d_loss > 100 or avg_ge_loss > 100:
            divergence_counter += 1
            print(f"WARNING: Training may be diverging (counter: {divergence_counter})")
            if divergence_counter >= 3:
                print("ERROR: Training has diverged! Stopping.")
                break
        else:
            divergence_counter = 0
        
        # Save checkpoints
        Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        
        # Save regular checkpoints
        torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_epoch{epoch}.pt")
        torch.save(E.state_dict(), f"{cfg['checkpoint_dir']}/E_epoch{epoch}.pt")
        torch.save(D.state_dict(), f"{cfg['checkpoint_dir']}/D_epoch{epoch}.pt")
        
        # Save EMA checkpoint if used
        if ema_G is not None:
            ema_G.apply_shadow()
            torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_ema_epoch{epoch}.pt")
            ema_G.restore()
        
        # Save best model
        if avg_d_loss < best_d_loss:
            best_d_loss = avg_d_loss
            torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_best.pt")
            torch.save(E.state_dict(), f"{cfg['checkpoint_dir']}/E_best.pt")
            torch.save(D.state_dict(), f"{cfg['checkpoint_dir']}/D_best.pt")
            print(f"Saved best model at epoch {epoch}")
        
        print(f"Saved checkpoints for epoch {epoch}\n")
        
        # Update learning rate
        if sched_G is not None:
            sched_G.step()
        if sched_D is not None:
            sched_D.step()
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved BiGAN training with stability features")
    
    # Data parameters
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    
    # Model parameters
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=2)
    
    # Optimization parameters
    parser.add_argument("--lr_ge", type=float, default=1e-4, help="Learning rate for Generator/Encoder")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Learning rate for Discriminator")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--lr_decay_epoch", type=int, default=15, help="Decay LR after this epoch (0=no decay)")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="LR decay factor")
    
    # Stability parameters
    parser.add_argument("--use_sn", action="store_true", default=False, help="Use spectral normalization")
    parser.add_argument("--enable_sn", action="store_true", dest="use_sn")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--n_critic", type=int, default=1, help="Train D this many times per G/E update")
    
    # EMA parameters
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA for generator")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_improved")
    
    args = parser.parse_args()
    cfg = vars(args)
    
    train(cfg)

