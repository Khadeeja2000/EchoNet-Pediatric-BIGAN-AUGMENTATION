"""
Step 2: Train BiGAN conditioned on Sex, Age, and BMI
Stable implementation with proper conditioning
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
from tqdm import tqdm


class EchoDataset(Dataset):
    def __init__(self, manifest_csv: str, frames: int = 32, size: int = 64):
        self.df = pd.read_csv(manifest_csv)
        # Filter valid processed videos
        self.df = self.df[self.df['processed_path'].notna()].reset_index(drop=True)
        self.frames = frames
        self.size = size
        
        # Create encoding mappings
        self.sex_map = {'F': 0, 'M': 1}
        self.age_map = {
            '0-1': 0, '2-5': 1, '6-10': 2, '11-15': 3, '16-18': 4
        }
        self.bmi_map = {
            'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3
        }
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"  Sex distribution: {dict(self.df['Sex'].value_counts())}")
        print(f"  Age distribution: {dict(self.df['age_bin'].value_counts())}")
        print(f"  BMI distribution: {dict(self.df['bmi_category'].value_counts())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['processed_path']
        
        # Encode conditions
        sex = self.sex_map.get(row['Sex'], 0)
        age = self.age_map.get(row['age_bin'], 0)
        bmi = self.bmi_map.get(row['bmi_category'], 0)
        
        # Condition vector: [sex, age, bmi]
        cond = torch.tensor([sex, age, bmi], dtype=torch.float32)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for _ in range(self.frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.frames:
            frames.append(frames[-1] if frames else np.zeros((self.size, self.size), dtype=np.uint8))
        
        # Convert to tensor: normalize to [-1, 1]
        arr = np.stack(frames[:self.frames], axis=0).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1, T, H, W]
        
        return x, cond


class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=3, channels=1, size=64):
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
            # 32 -> 64 (spatial only)
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

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4, 4)
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, z_dim=128, cond_dim=3, channels=1, size=64):
        super().__init__()
        layers = []
        
        if size >= 64:
            layers.extend([
                nn.Conv3d(channels, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        else:
            layers.extend([
                nn.Conv3d(channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
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

    def forward(self, x, c):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=3, channels=1, size=64):
        super().__init__()
        layers = []
        
        if size >= 64:
            layers.extend([
                nn.Conv3d(channels, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        else:
            layers.extend([
                nn.Conv3d(channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
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
        
        self.video_path = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4 + z_dim + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x, z, c):
        x = self.video_path(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, z, c], dim=1)
        return self.fc(x)


def train(cfg):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = EchoDataset(cfg['manifest'], frames=cfg['frames'], size=cfg['size'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        drop_last=True
    )
    
    # Models
    G = Generator(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    E = Encoder(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    D = Discriminator(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    
    # Optimizers - EQUAL learning rates for stability
    opt_G = optim.Adam(list(G.parameters()) + list(E.parameters()), lr=cfg['lr'], betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Z dimension: {cfg['z_dim']}")
    print(f"  Conditioning: Sex + Age (5 bins) + BMI (4 bins)")
    print()
    
    # Training loop
    for epoch in range(cfg['epochs']):
        G.train()
        E.train()
        D.train()
        
        epoch_d_losses = []
        epoch_ge_losses = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for i, (real_x, cond) in enumerate(pbar):
            real_x = real_x.to(device)
            cond = cond.to(device)
            batch_size = real_x.size(0)
            
            z = torch.randn(batch_size, cfg['z_dim'], device=device)
            
            # Train Discriminator
            opt_D.zero_grad()
            
            with torch.no_grad():
                enc_z = E(real_x, cond)
                fake_x = G(z, cond)
            
            real_score = D(real_x, enc_z, cond)
            fake_score = D(fake_x, z, cond)
            
            # Hinge loss for D
            d_loss = torch.relu(1.0 - real_score).mean() + torch.relu(1.0 + fake_score).mean()
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()
            
            # Train Generator + Encoder
            opt_G.zero_grad()
            
            fake_x = G(z, cond)
            enc_z = E(real_x, cond)
            
            fake_score = D(fake_x, z, cond)
            real_score = D(real_x, enc_z, cond)
            
            ge_loss = (-fake_score).mean() + (-real_score).mean()
            
            ge_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(G.parameters()) + list(E.parameters()), 1.0)
            opt_G.step()
            
            epoch_d_losses.append(d_loss.item())
            epoch_ge_losses.append(ge_loss.item())
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.3f}',
                'GE_loss': f'{ge_loss.item():.3f}'
            })
        
        # Epoch summary
        avg_d = np.mean(epoch_d_losses)
        avg_ge = np.mean(epoch_ge_losses)
        print(f"\nEpoch {epoch}: D_loss={avg_d:.4f}, GE_loss={avg_ge:.4f}")
        
        # Save checkpoints
        Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_epoch{epoch}.pt")
        torch.save(E.state_dict(), f"{cfg['checkpoint_dir']}/E_epoch{epoch}.pt")
        torch.save(D.state_dict(), f"{cfg['checkpoint_dir']}/D_epoch{epoch}.pt")
        print(f"Saved checkpoint for epoch {epoch}\n")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data_fresh/manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=3)  # sex, age, bmi
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_fresh")
    args = parser.parse_args()
    
    train(vars(args))

