import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2


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


def save_video(tensor: torch.Tensor, path: str, fps: int = 30) -> None:
    x = tensor.squeeze(0).detach().cpu().numpy()
    x = (x + 1.0) * 0.5
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    h, w = x.shape[1], x.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for t in range(x.shape[0]):
        frame = x[t]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/G_epoch0.pt")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="augmentation/samples")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = Generator(z_dim=args.z_dim, cond_dim=args.cond_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(state)
    G.eval()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sex_vals = [0.0, 1.0]
    age_bins = [0.0, 1.0, 2.0, 3.0, 4.0]
    cond_list = [(s, a) for s in sex_vals for a in age_bins]

    for idx in range(args.num_samples):
        sex, age = cond_list[idx % len(cond_list)]
        z = torch.randn(1, args.z_dim, device=device)
        c = torch.tensor([[sex, age]], dtype=torch.float32, device=device)
        with torch.no_grad():
            x = G(z, c)
        x_reordered = x[:, 0]
        out_path = os.path.join(args.out_dir, f"sample_{idx:03d}_s{int(sex)}_a{int(age)}.mp4")
        save_video(x_reordered, out_path, fps=args.fps)

    print(f"Saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
