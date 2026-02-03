"""
GENAI PROJECT: Video-to-Video Translation (A4C ↔ PSAX)
Uses CycleGAN architecture for conditional video translation
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


class ViewTranslationDataset(Dataset):
    """Dataset for A4C ↔ PSAX translation"""
    def __init__(self, manifest_csv: str, source_view: str, target_view: str, frames: int = 32, size: int = 64):
        self.df = pd.read_csv(manifest_csv)
        # Filter by source view
        self.df = self.df[self.df['view'] == source_view].reset_index(drop=True)
        self.source_view = source_view
        self.target_view = target_view
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


class VideoGenerator(nn.Module):
    """Generator for video translation"""
    def __init__(self, channels=1, frames=32, size=64):
        super().__init__()
        self.frames = frames
        self.size = size
        
        # Encoder - simpler architecture that preserves size
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.InstanceNorm3d(256),
            nn.ReLU(True),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock3D(256) for _ in range(6)]
        )
        
        # Decoder - same size preserving
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, channels, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(True),
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.InstanceNorm3d(channels),
        )
    
    def forward(self, x):
        return x + self.conv(x)


class VideoDiscriminator(nn.Module):
    """Discriminator for video"""
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, 4, 1, 1),
        )
    
    def forward(self, x):
        return self.net(x)


def train_cyclegan(cfg):
    """Train CycleGAN for view translation"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Datasets
    print("Loading datasets...")
    dataset_A = ViewTranslationDataset(cfg["manifest"], "A4C", "PSAX", cfg["frames"], cfg["size"])
    dataset_B = ViewTranslationDataset(cfg["manifest"], "PSAX", "A4C", cfg["frames"], cfg["size"])
    print(f"Dataset A (A4C): {len(dataset_A)} videos")
    print(f"Dataset B (PSAX): {len(dataset_B)} videos")
    
    loader_A = DataLoader(dataset_A, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    loader_B = DataLoader(dataset_B, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    
    # Models
    G_AB = VideoGenerator(cfg["channels"], cfg["frames"], cfg["size"]).to(device)  # A4C -> PSAX
    G_BA = VideoGenerator(cfg["channels"], cfg["frames"], cfg["size"]).to(device)  # PSAX -> A4C
    D_A = VideoDiscriminator(cfg["channels"]).to(device)  # Discriminator for A4C
    D_B = VideoDiscriminator(cfg["channels"]).to(device)  # Discriminator for PSAX
    
    # Optimizers
    opt_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=cfg["lr"], betas=(0.5, 0.999))
    
    # Loss functions
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    print("Starting training...")
    print(f"Dataset A (A4C): {len(dataset_A)} videos")
    print(f"Dataset B (PSAX): {len(dataset_B)} videos")
    
    for epoch in range(cfg["epochs"]):
        epoch_loss_G = 0
        epoch_loss_D = 0
        num_batches = 0
        
        # Use zip_longest to handle different dataset sizes
        from itertools import zip_longest
        for i, (real_A, real_B) in enumerate(zip_longest(loader_A, loader_B, fillvalue=None)):
            if real_A is None or real_B is None:
                break
                
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Train Generators
            opt_G.zero_grad()
            
            # Identity loss
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)
            loss_idt = criterion_identity(idt_A, real_A) + criterion_identity(idt_B, real_B)
            
            # GAN loss
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))
            
            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))
            
            # Cycle loss
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B) + 5.0 * loss_idt
            loss_G.backward()
            opt_G.step()
            
            # Train Discriminators
            opt_D.zero_grad()
            
            # Discriminator A
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_gan(pred_real_A, torch.ones_like(pred_real_A))
            pred_fake_A = D_A(fake_A.detach())
            loss_D_fake_A = criterion_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            
            # Discriminator B
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_gan(pred_real_B, torch.ones_like(pred_real_B))
            pred_fake_B = D_B(fake_B.detach())
            loss_D_fake_B = criterion_gan(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")
        
        avg_loss_G = epoch_loss_G / num_batches if num_batches > 0 else 0
        avg_loss_D = epoch_loss_D / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{cfg['epochs']} - Avg Loss G: {avg_loss_G:.4f}, Avg Loss D: {avg_loss_D:.4f}")
        
        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch{epoch+1}.pt")
            torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch{epoch+1}.pt")
            print(f"✓ Saved checkpoints at epoch {epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--frames", type=int, default=32)
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
    
    train_cyclegan(cfg)

