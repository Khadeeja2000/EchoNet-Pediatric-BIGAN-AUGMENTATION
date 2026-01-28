"""
Improved sample generation script with quality validation.
Generates synthetic echocardiogram videos using trained BiGAN generator.
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
    def __init__(self, z_dim: int = 128, cond_dim: int = 2, channels: int = 1, size: int = 64):
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
            # 32 -> 64 (spatial-only if needed)
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

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4, 4)
        return self.net(x)


def save_video(tensor: torch.Tensor, path: str, fps: int = 30) -> None:
    """Save video tensor to file"""
    # tensor shape: [1, C, T, H, W]
    x = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()  # [T, H, W]
    
    # Denormalize from [-1, 1] to [0, 255]
    x = (x + 1.0) * 0.5
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    
    T, H, W = x.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=False)
    
    for t in range(T):
        frame = x[t]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()


def validate_video_quality(tensor: torch.Tensor) -> dict:
    """Basic quality checks for generated video"""
    x = tensor.squeeze().detach().cpu().numpy()
    
    metrics = {
        'mean': float(x.mean()),
        'std': float(x.std()),
        'min': float(x.min()),
        'max': float(x.max()),
        'has_nan': bool(np.isnan(x).any()),
        'has_inf': bool(np.isinf(x).any()),
        'temporal_variance': float(x.var(axis=0).mean()),  # Variance across time
        'spatial_variance': float(x.var(axis=(1, 2)).mean()),  # Variance across space
    }
    
    # Check if video is reasonable
    metrics['is_valid'] = (
        not metrics['has_nan'] and 
        not metrics['has_inf'] and 
        metrics['std'] > 0.01 and  # Not constant
        metrics['temporal_variance'] > 0.001  # Has temporal variation
    )
    
    return metrics


def generate_samples(
    generator: nn.Module,
    z_dim: int,
    num_samples: int,
    out_dir: str,
    device: str,
    fps: int = 30,
    validate: bool = True
) -> None:
    """Generate video samples with all condition combinations"""
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Define conditions: sex (F=0, M=1) × age_bins (0-1, 2-5, 6-10, 11-15, 16-18)
    sex_labels = {0: 'F', 1: 'M'}
    age_labels = {0: '0-1', 1: '2-5', 2: '6-10', 3: '11-15', 4: '16-18'}
    
    conditions = []
    for sex in [0, 1]:
        for age in [0, 1, 2, 3, 4]:
            conditions.append((sex, age))
    
    generator.eval()
    valid_count = 0
    invalid_count = 0
    
    print(f"\nGenerating {num_samples} samples...")
    print(f"Output directory: {out_dir}\n")
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Generating videos"):
            # Cycle through conditions
            sex, age = conditions[idx % len(conditions)]
            
            # Sample random latent vector
            z = torch.randn(1, z_dim, device=device)
            c = torch.tensor([[sex, age]], dtype=torch.float32, device=device)
            
            # Generate video
            video = generator(z, c)
            
            # Validate quality
            if validate:
                metrics = validate_video_quality(video)
                if not metrics['is_valid']:
                    print(f"\n⚠️  Warning: Sample {idx} may have quality issues:")
                    print(f"    mean={metrics['mean']:.3f}, std={metrics['std']:.3f}")
                    print(f"    has_nan={metrics['has_nan']}, has_inf={metrics['has_inf']}")
                    invalid_count += 1
                else:
                    valid_count += 1
            
            # Save video
            sex_label = sex_labels[sex]
            age_label = age_labels[age]
            filename = f"sample_{idx:03d}_sex{sex_label}_age{age_label}.mp4"
            filepath = os.path.join(out_dir, filename)
            
            save_video(video, filepath, fps=fps)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    if validate:
        print(f"Valid samples: {valid_count}")
        print(f"Invalid samples: {invalid_count}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic echocardiogram videos")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--cond_dim", type=int, default=2, help="Condition dimension")
    parser.add_argument("--size", type=int, default=64, help="Video spatial size")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--out_dir", type=str, default="augmentation/generated_samples", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--no_validate", action="store_true", help="Skip quality validation")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"\nUsing device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load generator
    try:
        G = Generator(z_dim=args.z_dim, cond_dim=args.cond_dim, size=args.size).to(device)
        state = torch.load(args.checkpoint, map_location=device)
        G.load_state_dict(state)
        print("✓ Generator loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading generator: {e}")
        return
    
    # Generate samples
    try:
        generate_samples(
            generator=G,
            z_dim=args.z_dim,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
            device=device,
            fps=args.fps,
            validate=not args.no_validate
        )
    except Exception as e:
        print(f"❌ ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("✅ Done!")


if __name__ == "__main__":
    main()

