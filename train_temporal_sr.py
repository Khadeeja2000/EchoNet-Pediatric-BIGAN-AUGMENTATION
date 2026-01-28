"""
COMPUTER VISION PROJECT: Temporal Super-Resolution (Frame Interpolation)
Uses optical flow and neural networks to increase frame rate
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


class TemporalSRDataset(Dataset):
    """Dataset for temporal super-resolution (frame interpolation)"""
    def __init__(self, manifest_csv: str, frames: int = 32, size: int = 64, scale: int = 2):
        self.df = pd.read_csv(manifest_csv)
        if "processed_path" in self.df.columns:
            self.df = self.df[self.df["processed_path"].astype(str).str.len() > 0].reset_index(drop=True)
        self.frames = frames  # Input frames
        self.target_frames = frames * scale  # Output frames (2x)
        self.size = size
        self.scale = scale
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["processed_path"] if "processed_path" in row else row["file_path"]
        
        # Load video with MORE frames (for training)
        cap = cv2.VideoCapture(video_path)
        frames_list = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.target_frames
        
        # Sample target_frames frames
        indices = np.linspace(0, max(total - 1, 0), self.target_frames).astype(int)
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
            frames_list = [np.zeros((self.size, self.size), dtype=np.uint8) for _ in range(self.target_frames)]
        if len(frames_list) < self.target_frames:
            frames_list += [frames_list[-1]] * (self.target_frames - len(frames_list))
        
        video_full = np.stack(frames_list[:self.target_frames], axis=0).astype(np.float32) / 255.0
        
        # Downsample to input frames (every other frame)
        video_input = video_full[::self.scale]  # Take every 2nd frame
        
        return (torch.from_numpy(video_input).unsqueeze(0),  # Input: (1, T, H, W)
                torch.from_numpy(video_full).unsqueeze(0))    # Target: (1, 2T, H, W)


class FrameInterpolationNet(nn.Module):
    """Neural network for frame interpolation"""
    def __init__(self, channels=1, size=64):
        super().__init__()
        self.size = size
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 64, (3, 3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv3d(64, 128, (3, 3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv3d(128, 256, (3, 3, 3), 1, 1),
            nn.ReLU(True),
        )
        
        # Temporal upsampling
        self.temporal_upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2, 1, 1), (2, 1, 1), 0),  # 2x temporal upsampling
            nn.ReLU(True),
            nn.Conv3d(128, 128, (3, 3, 3), 1, 1),
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, (3, 3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv3d(64, channels, (3, 3, 3), 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: (B, 1, T, H, W)
        x = self.encoder(x)
        x = self.temporal_upsample(x)
        x = self.decoder(x)
        return x


def train_temporal_sr(cfg):
    """Train temporal super-resolution model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = TemporalSRDataset(cfg["manifest"], cfg["frames"], cfg["size"], scale=2)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    
    model = FrameInterpolationNet(cfg["channels"], cfg["size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(cfg["epochs"]):
        for i, (input_frames, target_frames) in enumerate(loader):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)
            
            optimizer.zero_grad()
            
            # Predict interpolated frames
            output_frames = model(input_frames)
            
            # Loss
            loss = criterion(output_frames, target_frames)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/temporal_sr_epoch{epoch+1}.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--frames", type=int, default=16)  # Input frames
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
    
    train_temporal_sr(cfg)





