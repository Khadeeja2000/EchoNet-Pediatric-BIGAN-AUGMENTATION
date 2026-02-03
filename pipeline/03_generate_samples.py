"""
Step 3: Generate augmented samples conditioned on Sex, Age, and BMI
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=3, channels=1, size=64):
        super().__init__()
        self.size = size
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


def save_video(tensor, path, fps=30):
    """Save tensor as video file"""
    x = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()  # [T, H, W]
    
    # Denormalize from [-1, 1] to [0, 255]
    x = (x + 1.0) * 0.5
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    
    T, H, W = x.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=False)
    
    for t in range(T):
        frame = x[t]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()


def generate_samples(checkpoint, num_samples, out_dir, device, z_dim=128, size=64):
    """Generate samples with all combinations of conditions"""
    
    # Load generator
    G = Generator(z_dim=z_dim, cond_dim=3, size=size).to(device)
    G.load_state_dict(torch.load(checkpoint, map_location=device))
    G.eval()
    
    print(f"✓ Loaded generator from {checkpoint}")
    
    # Define condition mappings
    sex_labels = {0: 'F', 1: 'M'}
    age_labels = {0: '0-1', 1: '2-5', 2: '6-10', 3: '11-15', 4: '16-18'}
    bmi_labels = {0: 'underweight', 1: 'normal', 2: 'overweight', 3: 'obese'}
    
    # Create all combinations
    conditions = []
    for sex in [0, 1]:
        for age in [0, 1, 2, 3, 4]:
            for bmi in [0, 1, 2, 3]:
                conditions.append((sex, age, bmi))
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Conditioning on {len(conditions)} unique combinations (Sex × Age × BMI)")
    print(f"Output directory: {out_dir}\n")
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Generating"):
            # Cycle through condition combinations
            sex, age, bmi = conditions[idx % len(conditions)]
            
            # Sample random latent
            z = torch.randn(1, z_dim, device=device)
            c = torch.tensor([[sex, age, bmi]], dtype=torch.float32, device=device)
            
            # Generate
            video = G(z, c)
            
            # Create filename
            sex_label = sex_labels[sex]
            age_label = age_labels[age]
            bmi_label = bmi_labels[bmi]
            filename = f"sample_{idx:04d}_sex{sex_label}_age{age_label}_bmi{bmi_label}.mp4"
            filepath = os.path.join(out_dir, filename)
            
            # Save
            save_video(video, filepath)
    
    print(f"\n✅ Generated {num_samples} samples in {out_dir}")
    
    # Summary
    print(f"\nCondition mappings:")
    print(f"  Sex: F=0, M=1")
    print(f"  Age: {age_labels}")
    print(f"  BMI: {bmi_labels}")


def main():
    parser = argparse.ArgumentParser(description="Generate conditioned synthetic echocardiograms")
    parser.add_argument("--checkpoint", type=str, required=True, help="Generator checkpoint path")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--out_dir", type=str, default="augmented_data", help="Output directory")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--size", type=int, default=64, help="Video resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Generate
    generate_samples(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        device=device,
        z_dim=args.z_dim,
        size=args.size
    )


if __name__ == "__main__":
    main()

