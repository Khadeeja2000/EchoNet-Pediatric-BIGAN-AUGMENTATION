"""
Generate translated videos using trained CycleGAN model
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import cv2
from train_view_translation import VideoGenerator, ViewTranslationDataset
from torch.utils.data import DataLoader


def save_video(tensor, path, fps=30):
    """Save video tensor to file"""
    x = tensor.squeeze(0).detach().cpu().numpy()  # (T, H, W)
    x = (x + 1.0) * 127.5  # [-1,1] -> [0,255]
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    h, w = x.shape[1], x.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for t in range(x.shape[0]):
        frame = x[t]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved: {path}")


def generate_samples(cfg):
    """Generate translated video samples"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    G_AB = VideoGenerator(cfg["channels"], cfg["frames"], cfg["size"]).to(device)  # A4C -> PSAX
    G_BA = VideoGenerator(cfg["channels"], cfg["frames"], cfg["size"]).to(device)  # PSAX -> A4C
    
    # Load checkpoints
    if os.path.exists(cfg["checkpoint_AB"]):
        G_AB.load_state_dict(torch.load(cfg["checkpoint_AB"], map_location=device))
        print(f"Loaded: {cfg['checkpoint_AB']}")
    else:
        print(f"Warning: {cfg['checkpoint_AB']} not found!")
        return
    
    if os.path.exists(cfg["checkpoint_BA"]):
        G_BA.load_state_dict(torch.load(cfg["checkpoint_BA"], map_location=device))
        print(f"Loaded: {cfg['checkpoint_BA']}")
    
    G_AB.eval()
    G_BA.eval()
    
    # Load sample videos
    dataset_A = ViewTranslationDataset(cfg["manifest"], "A4C", "PSAX", cfg["frames"], cfg["size"])
    dataset_B = ViewTranslationDataset(cfg["manifest"], "PSAX", "A4C", cfg["frames"], cfg["size"])
    
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Generate A4C -> PSAX
    print("\nGenerating A4C -> PSAX translations...")
    for i in range(min(cfg["num_samples"], len(dataset_A))):
        real_A = dataset_A[i].unsqueeze(0).to(device)
        
        with torch.no_grad():
            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
        
        # Save original A4C
        save_video(real_A, os.path.join(cfg["output_dir"], f"original_A4C_{i:03d}.mp4"))
        # Save translated PSAX
        save_video(fake_B, os.path.join(cfg["output_dir"], f"translated_A4C_to_PSAX_{i:03d}.mp4"))
        # Save reconstructed A4C (cycle)
        save_video(rec_A, os.path.join(cfg["output_dir"], f"reconstructed_A4C_{i:03d}.mp4"))
    
    # Generate PSAX -> A4C
    if os.path.exists(cfg["checkpoint_BA"]):
        print("\nGenerating PSAX -> A4C translations...")
        for i in range(min(cfg["num_samples"], len(dataset_B))):
            real_B = dataset_B[i].unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake_A = G_BA(real_B)
                rec_B = G_AB(fake_A)
            
            # Save original PSAX
            save_video(real_B, os.path.join(cfg["output_dir"], f"original_PSAX_{i:03d}.mp4"))
            # Save translated A4C
            save_video(fake_A, os.path.join(cfg["output_dir"], f"translated_PSAX_to_A4C_{i:03d}.mp4"))
            # Save reconstructed PSAX (cycle)
            save_video(rec_B, os.path.join(cfg["output_dir"], f"reconstructed_PSAX_{i:03d}.mp4"))
    
    print(f"\n✓ Generated {cfg['num_samples']} samples in {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--checkpoint_AB", type=str, default="checkpoints/G_AB_epoch10.pt")
    parser.add_argument("--checkpoint_BA", type=str, default="checkpoints/G_BA_epoch10.pt")
    parser.add_argument("--output_dir", type=str, default="view_translation_samples")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()
    
    cfg = {
        "manifest": args.manifest,
        "checkpoint_AB": args.checkpoint_AB,
        "checkpoint_BA": args.checkpoint_BA,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "frames": args.frames,
        "size": args.size,
        "channels": args.channels,
    }
    
    generate_samples(cfg)

