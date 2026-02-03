import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import cv2


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, cond_dim: int = 2, channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 256 * 4 * 4 * 4)
        
        self.net = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose3d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 256, 4, 4, 4)
        return self.net(x)


def save_video(tensor: torch.Tensor, path: str, fps: int = 30) -> None:
    # tensor: (1, T, H, W) in [-1,1]
    x = tensor.squeeze(0).detach().cpu().numpy()  # (T, H, W)
    x = (x + 1.0) * 0.5  # [0,1]
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    
    h, w = x.shape[1], x.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for t in range(x.shape[0]):
        frame = x[t]
        # Convert grayscale to BGR for mp4
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stable_G_epoch19.pt")
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--cond_dim", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="augmentation/stable_samples")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load generator
    G = Generator(z_dim=args.z_dim, cond_dim=args.cond_dim).to(device)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        G.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: checkpoint {args.checkpoint} not found, using random weights")
    G.eval()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Generate samples for different conditions
    sex_vals = [0.0, 1.0]  # Female, Male
    age_bins = [0.0, 1.0, 2.0, 3.0, 4.0]  # 0-1, 2-5, 6-10, 11-15, 16-18

    idx = 0
    samples_per_condition = max(1, args.num_samples // (len(sex_vals) * len(age_bins)))
    
    print(f"Generating {args.num_samples} samples...")
    for sex in sex_vals:
        for age in age_bins:
            for _ in range(samples_per_condition):
                if idx >= args.num_samples:
                    break
                
                # Sample random z
                z = torch.randn(1, args.z_dim, device=device)
                
                # Create condition
                c = torch.tensor([[sex, age]], dtype=torch.float32, device=device)
                
                # Generate
                with torch.no_grad():
                    x = G(z, c)  # (1, 1, T, H, W)
                
                # Save
                sex_str = "F" if sex == 0.0 else "M"
                age_str = ["0-1", "2-5", "6-10", "11-15", "16-18"][int(age)]
                out_path = os.path.join(args.out_dir, f"sample_{idx:03d}_sex{sex_str}_age{age_str}.mp4")
                save_video(x[:, 0], out_path, fps=args.fps)
                
                idx += 1
            if idx >= args.num_samples:
                break
        if idx >= args.num_samples:
            break

    print(f"\nGenerated {idx} samples in {args.out_dir}")


if __name__ == "__main__":
    main()
