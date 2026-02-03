"""
GENAI PROJECT: Video Diffusion Model for Echocardiogram Generation
Uses diffusion process to generate videos - modern GenAI approach
Different from GAN - uses denoising diffusion
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
import math


class EchoVideoDataset(Dataset):
    """Dataset for video diffusion training"""
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
        return torch.from_numpy(video).unsqueeze(0)  # (1, T, H, W)


class SinusoidalPositionalEmbedding(nn.Module):
    """Positional embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class VideoDiffusionModel(nn.Module):
    """U-Net based video diffusion model"""
    def __init__(self, channels=1, frames=32, size=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embed = SinusoidalPositionalEmbedding(time_emb_dim)
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(channels, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Middle
        self.mid = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, 3, 2, 1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, 3, 2, 1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(128, channels, 3, 1, 1),
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        t_emb = t_emb.view(-1, 256, 1, 1, 1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Middle with time conditioning
        mid = self.mid(e3)
        mid = mid + t_emb
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([mid, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1


def linear_beta_schedule(timesteps):
    """Linear noise schedule"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """Extract values from schedule at timestep t"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def train_diffusion(cfg):
    """Train video diffusion model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset
    dataset = EchoVideoDataset(cfg["manifest"], cfg["frames"], cfg["size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    
    # Model
    model = VideoDiffusionModel(cfg["channels"], cfg["frames"], cfg["size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    
    # Diffusion schedule
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print(f"Dataset: {len(dataset)} videos")
    print(f"Starting training for {cfg['epochs']} epochs...")
    
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, video in enumerate(loader):
            video = video.to(device)
            batch_size = video.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Sample noise
            noise = torch.randn_like(video)
            
            # Add noise to video
            sqrt_alphas_cumprod_t = extract(alphas_cumprod.sqrt(), t, video.shape)
            sqrt_one_minus_alphas_cumprod_t = extract((1.0 - alphas_cumprod).sqrt(), t, video.shape)
            noisy_video = sqrt_alphas_cumprod_t * video + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Predict noise
            optimizer.zero_grad()
            predicted_noise = model(noisy_video, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{cfg['epochs']} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/diffusion_epoch{epoch+1}.pt")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")


@torch.no_grad()
def sample_diffusion(model, shape, device, timesteps=1000):
    """Generate video using diffusion process"""
    model.eval()
    
    # Noise schedule
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start from pure noise
    video = torch.randn(shape, device=device)
    
    # Reverse diffusion process
    for i in reversed(range(0, timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(video, t)
        
        # Denoise
        alpha_t = alphas_cumprod[i]
        alpha_t_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0)
        
        pred_video = (1 / alpha_t.sqrt()) * (video - (1 - alpha_t) / (1 - alphas_cumprod[i]).sqrt() * predicted_noise)
        
        if i > 0:
            noise = torch.randn_like(video)
            pred_video = pred_video + ((1 - alpha_t_prev) / (1 - alpha_t)).sqrt() * betas[i] * noise
        
        video = pred_video
    
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()
    
    cfg = {
        "manifest": args.manifest,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "frames": args.frames,
        "size": args.size,
        "channels": args.channels,
    }
    
    train_diffusion(cfg)




