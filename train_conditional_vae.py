"""
GENAI PROJECT: Conditional VAE for Video Generation
Simple and fast - generates videos conditioned on patient demographics
Much faster than diffusion or complex GANs
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2


class ConditionalVideoDataset(Dataset):
    """Dataset with conditions (sex, age)"""
    def __init__(self, manifest_csv: str, frames: int = 16, size: int = 64):
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
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames_list = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.frames
        indices = np.linspace(0, max(total - 1, 0), self.frames).astype(int)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            f = cv2.resize(f, (self.size, self.size))
            frames_list.append(f)
        cap.release()
        
        if len(frames_list) == 0:
            frames_list = [np.zeros((self.size, self.size), dtype=np.uint8) for _ in range(self.frames)]
        if len(frames_list) < self.frames:
            frames_list += [frames_list[-1]] * (self.frames - len(frames_list))
        
        video = np.stack(frames_list[:self.frames], axis=0).astype(np.float32) / 255.0
        
        # Get conditions
        sex_val = str(row["sex"]).strip().lower()
        sex = 0.0 if sex_val.startswith("f") else 1.0
        age_bin = row["age_bin"] if "age_bin" in row else "other"
        age_mapping = {"0-1": 0.0, "2-5": 1.0, "6-10": 2.0, "11-15": 3.0, "16-18": 4.0, "other": 5.0}
        age = age_mapping.get(str(age_bin), 5.0) / 5.0  # Normalize to [0,1]
        
        condition = torch.tensor([sex, age], dtype=torch.float32)
        
        return torch.from_numpy(video).unsqueeze(0), condition  # (1, T, H, W), (2,)


class ConditionalVAE(nn.Module):
    """Simple Conditional VAE for fast training"""
    def __init__(self, z_dim=64, cond_dim=2, channels=1, frames=16, size=64):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 32, 3, 2, 1),  # 16x64x64 -> 8x32x32
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, 2, 1),  # 8x32x32 -> 4x16x16
            nn.ReLU(True),
            nn.Conv3d(64, 128, 3, 2, 1),  # 4x16x16 -> 2x8x8
            nn.ReLU(True),
        )
        
        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, 128)
        
        # Latent space
        self.fc_mu = nn.Linear(128 * 2 * 8 * 8 + 128, z_dim)
        self.fc_logvar = nn.Linear(128 * 2 * 8 * 8 + 128, z_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(z_dim + cond_dim, 128 * 2 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, output_padding=1),  # 2x8x8 -> 4x16x16
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1),  # 4x16x16 -> 8x32x32
            nn.ReLU(True),
            nn.ConvTranspose3d(32, channels, 3, 2, 1, output_padding=1),  # 8x32x32 -> 16x64x64
            nn.Sigmoid(),
        )
    
    def encode(self, x, c):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        c_proj = self.cond_proj(c)
        h = torch.cat([h, c_proj], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.fc_decode(h)
        h = h.view(h.size(0), 128, 2, 8, 8)
        return self.decoder(h)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


def train_cvae(cfg):
    """Train Conditional VAE"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = ConditionalVideoDataset(cfg["manifest"], cfg["frames"], cfg["size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    
    model = ConditionalVAE(cfg["z_dim"], cfg["cond_dim"], cfg["channels"], cfg["frames"], cfg["size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    
    print(f"Dataset: {len(dataset)} videos")
    print(f"Starting training for {cfg['epochs']} epochs...")
    print("This will be FAST - VAE trains much quicker than diffusion!")
    
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (video, condition) in enumerate(loader):
            video = video.to(device)
            condition = condition.to(device)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(video, condition)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, video)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / video.numel()
            
            loss = recon_loss + 0.0001 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{cfg['epochs']} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint every epoch (fast!)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/cvae_epoch{epoch+1}.pt")
        print(f"✓ Saved checkpoint at epoch {epoch+1}")


@torch.no_grad()
def generate_cvae(model, condition, device, z_dim=64):
    """Generate video from condition"""
    model.eval()
    z = torch.randn(1, z_dim, device=device)
    c = condition.unsqueeze(0).to(device)
    generated = model.decode(z, c)
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=2)
    args = parser.parse_args()
    
    cfg = {
        "manifest": args.manifest,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "frames": args.frames,
        "size": args.size,
        "channels": args.channels,
        "z_dim": args.z_dim,
        "cond_dim": args.cond_dim,
    }
    
    train_cvae(cfg)




