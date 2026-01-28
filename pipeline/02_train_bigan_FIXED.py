"""
Step 2: Train BiGAN - FIXED VERSION
- Loads numpy arrays (no codec issues)
- Fixed discriminator collapse (different loss function)
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
from tqdm import tqdm


class EchoDatasetNumpy(Dataset):
    """Dataset that loads from numpy arrays"""
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['processed_path'].notna()].reset_index(drop=True)
        
        # Encodings
        self.sex_map = {'F': 0, 'M': 1, 'O': 0}  # Map O to F for simplicity
        self.age_map = {'0-1': 0, '2-5': 1, '6-10': 2, '11-15': 3, '16-18': 4}
        self.bmi_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
        
        print(f"Dataset loaded: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load numpy array
        video = np.load(row['processed_path'])  # [T, H, W], uint8
        
        # Normalize to [-1, 1]
        video = video.astype(np.float32) / 127.5 - 1.0
        
        # To tensor [1, T, H, W]
        x = torch.from_numpy(video).unsqueeze(0)
        
        # Encode conditions
        sex = self.sex_map.get(row['Sex'], 0)
        age = self.age_map.get(row['age_bin'], 0)
        bmi = self.bmi_map.get(row['bmi_category'], 0)
        
        cond = torch.tensor([sex, age, bmi], dtype=torch.float32)
        
        return x, cond


class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=3, channels=1, size=64):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)
        
        layers = []
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        self.video_path = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4 + z_dim + cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output [0, 1] for BCE loss
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
    dataset = EchoDatasetNumpy(cfg['manifest'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 for Mac compatibility
        drop_last=True
    )
    
    # Models
    G = Generator(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    E = Encoder(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    D = Discriminator(cfg['z_dim'], cfg['cond_dim'], size=cfg['size']).to(device)
    
    # Optimizers - same LR for stability
    opt_G = optim.Adam(list(G.parameters()) + list(E.parameters()), lr=cfg['lr'], betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Loss: Binary Cross-Entropy (more stable)")
    print(f"  Conditioning: Sex + Age + BMI")
    print()
    
    # Training loop
    for epoch in range(cfg['epochs']):
        G.train()
        E.train()
        D.train()
        
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_e_losses = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for real_x, cond in pbar:
            real_x = real_x.to(device)
            cond = cond.to(device)
            batch_size = real_x.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            z = torch.randn(batch_size, cfg['z_dim'], device=device)
            
            # ===== Train Discriminator =====
            opt_D.zero_grad()
            
            # Real: (real_x, E(real_x)) should be real
            enc_z = E(real_x, cond).detach()
            real_pred = D(real_x, enc_z, cond)
            d_loss_real = criterion(real_pred, real_labels)
            
            # Fake: (G(z), z) should be fake
            fake_x = G(z, cond).detach()
            fake_pred = D(fake_x, z, cond)
            d_loss_fake = criterion(fake_pred, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()
            
            # ===== Train Generator + Encoder =====
            opt_G.zero_grad()
            
            # Generator: D(G(z), z) should look real
            fake_x = G(z, cond)
            fake_pred = D(fake_x, z, cond)
            g_loss = criterion(fake_pred, real_labels)  # Fool D
            
            # Encoder: D(real_x, E(real_x)) should look real
            enc_z = E(real_x, cond)
            real_pred = D(real_x, enc_z, cond)
            e_loss = criterion(real_pred, real_labels)  # Fool D
            
            ge_loss = g_loss + e_loss
            ge_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(G.parameters()) + list(E.parameters()), 1.0)
            opt_G.step()
            
            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss.item())
            epoch_e_losses.append(e_loss.item())
            
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss.item():.3f}',
                'E': f'{e_loss.item():.3f}'
            })
        
        # Epoch summary
        print(f"\nEpoch {epoch}: D={np.mean(epoch_d_losses):.4f}, "
              f"G={np.mean(epoch_g_losses):.4f}, E={np.mean(epoch_e_losses):.4f}")
        
        # Save checkpoints
        Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_epoch{epoch}.pt")
        torch.save(E.state_dict(), f"{cfg['checkpoint_dir']}/E_epoch{epoch}.pt")
        torch.save(D.state_dict(), f"{cfg['checkpoint_dir']}/D_epoch{epoch}.pt")
        print(f"Saved checkpoint\n")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data_numpy/manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_numpy")
    args = parser.parse_args()
    
    train(vars(args))

