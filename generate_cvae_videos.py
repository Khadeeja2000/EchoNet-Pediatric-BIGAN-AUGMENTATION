"""
Generate videos using trained Conditional VAE
Fast generation - just sample from latent space
"""
import argparse
import os
import numpy as np
import torch
import cv2
from train_conditional_vae import ConditionalVAE, generate_cvae


def save_video(tensor, path, fps=30):
    """Save video tensor to file"""
    x = tensor.squeeze(0).detach().cpu().numpy()  # (T, H, W)
    x = np.clip(x, 0, 1) * 255.0
    x = x.astype(np.uint8)
    
    h, w = x.shape[1], x.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for t in range(x.shape[0]):
        frame = x[t]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"  ✓ Saved: {path}")


def generate_samples(cfg):
    """Generate video samples"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = ConditionalVAE(cfg["z_dim"], cfg["cond_dim"], cfg["channels"], cfg["frames"], cfg["size"]).to(device)
    
    if os.path.exists(cfg["checkpoint"]):
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
        print(f"✓ Loaded: {cfg['checkpoint']}")
    else:
        print(f"⚠ Checkpoint not found: {cfg['checkpoint']}")
        return
    
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Generate videos with different conditions
    print(f"\nGenerating {cfg['num_samples']} videos...")
    
    conditions = [
        (0.0, 0.0),  # Female, 0-1 years
        (0.0, 0.5),  # Female, 6-10 years
        (1.0, 0.5),  # Male, 6-10 years
        (1.0, 1.0),  # Male, 16-18 years
        (0.0, 0.8),  # Female, 11-15 years
    ]
    
    for i, (sex, age) in enumerate(conditions[:cfg["num_samples"]]):
        print(f"  Generating video {i+1}/{cfg['num_samples']} (Sex: {'F' if sex==0 else 'M'}, Age: {age:.1f})...")
        condition = torch.tensor([sex, age], dtype=torch.float32)
        generated_video = generate_cvae(model, condition, device, cfg["z_dim"])
        
        sex_str = "F" if sex == 0 else "M"
        age_str = f"{age:.1f}"
        output_path = os.path.join(cfg["output_dir"], f"cvae_generated_sex{sex_str}_age{age_str}_{i:03d}.mp4")
        save_video(generated_video[0], output_path)
    
    print(f"\n✓ Generated {cfg['num_samples']} videos in {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cvae_epoch5.pt")
    parser.add_argument("--output_dir", type=str, default="cvae_generated")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=2)
    args = parser.parse_args()
    
    cfg = {
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "frames": args.frames,
        "size": args.size,
        "channels": args.channels,
        "z_dim": args.z_dim,
        "cond_dim": args.cond_dim,
    }
    
    generate_samples(cfg)




