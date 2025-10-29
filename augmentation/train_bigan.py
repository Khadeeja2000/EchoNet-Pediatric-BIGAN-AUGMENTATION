import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import cv2


class EchoDataset(Dataset):
    def __init__(self, manifest_csv: str, frames: int = 32, size: int = 128):
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

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        x = torch.from_numpy(arr).unsqueeze(0)
        return x, cond


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 256 * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 256, 4, 4, 4)
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4 * 4 + cond_dim, z_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.net(x).view(b, -1)
        x = torch.cat([x, c], dim=1)
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1):
        super().__init__()
        self.video_path = nn.Sequential(
            nn.Conv3d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4 + z_dim + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.video_path(x).view(b, -1)
        x = torch.cat([x, z, c], dim=1)
        return self.fc(x)


def gradient_penalty(D: nn.Module, real_x: torch.Tensor, fake_x: torch.Tensor, z: torch.Tensor, c: torch.Tensor, device: str) -> torch.Tensor:
    alpha = torch.rand(real_x.size(0), 1, 1, 1, 1, device=device)
    interpolates = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    d_interpolates = D(interpolates, z, c)
    grad = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train(cfg: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EchoDataset(cfg["manifest"], frames=cfg["frames"], size=cfg["size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    G = Generator(cfg["z_dim"], cfg["cond_dim"]).to(device)
    E = Encoder(cfg["z_dim"], cfg["cond_dim"]).to(device)
    D = Discriminator(cfg["z_dim"], cfg["cond_dim"]).to(device)

    opt_G = optim.Adam(list(G.parameters()) + list(E.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(cfg["epochs"]):
        for i, (real_x, cond) in enumerate(loader):
            real_x = real_x.to(device)
            cond = cond.to(device)
            z = torch.randn(real_x.size(0), cfg["z_dim"], device=device)

            opt_D.zero_grad()
            fake_x = G(z, cond)
            enc_z = E(real_x, cond)

            real_score = D(real_x, enc_z.detach(), cond)
            fake_score = D(fake_x.detach(), z.detach(), cond)
            gp = gradient_penalty(D, real_x, fake_x, z, cond, device)

            loss_D = -(torch.mean(real_score) - torch.mean(fake_score)) + cfg["lambda_gp"] * gp
            loss_D.backward()
            opt_D.step()

            if i % cfg["n_critic"] == 0:
                opt_G.zero_grad()
                fake_x = G(z, cond)
                enc_z = E(real_x, cond)
                fake_score = D(fake_x, z, cond)
                real_score = D(real_x, enc_z, cond)
                loss_GE = -(torch.mean(fake_score) + torch.mean(real_score))
                loss_GE.backward()
                opt_G.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{cfg['epochs']}] [Batch {i}/{len(loader)}] Loss_D: {loss_D.item():.4f}")

        Path("augmentation/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(G.state_dict(), f"augmentation/checkpoints/G_epoch{epoch}.pt")
        torch.save(E.state_dict(), f"augmentation/checkpoints/E_epoch{epoch}.pt")
        torch.save(D.state_dict(), f"augmentation/checkpoints/D_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=2)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--n_critic", type=int, default=5)
    args = parser.parse_args()

    cfg = vars(args)
    train(cfg)
