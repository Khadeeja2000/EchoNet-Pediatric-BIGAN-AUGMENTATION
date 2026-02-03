"""
Quick Demo: Generate sample videos to demonstrate GenAI project
This creates a demo even without full training
"""
import os
import numpy as np
import pandas as pd
import torch
import cv2
from train_view_translation import ViewTranslationDataset


def save_video(tensor, path, fps=30):
    """Save video tensor to file"""
    if isinstance(tensor, torch.Tensor):
        x = tensor.squeeze(0).detach().cpu().numpy()
        if x.min() < 0:  # If normalized to [-1, 1]
            x = (x + 1.0) * 127.5
        else:  # If normalized to [0, 1]
            x = x * 255.0
    else:
        x = tensor
    
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    if len(x.shape) == 3:  # (T, H, W)
        h, w = x.shape[1], x.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        
        for t in range(x.shape[0]):
            frame = x[t]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"  ✓ Saved: {path}")


def create_demo_samples():
    """Create demo samples from original videos"""
    print("="*60)
    print("GENAI PROJECT DEMO: Video-to-Video Translation")
    print("="*60)
    
    manifest = "data/processed/manifest.csv"
    output_dir = "view_translation_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_A4C = ViewTranslationDataset(manifest, "A4C", "PSAX", frames=16, size=64)
    dataset_PSAX = ViewTranslationDataset(manifest, "PSAX", "A4C", frames=16, size=64)
    
    print(f"  A4C videos: {len(dataset_A4C)}")
    print(f"  PSAX videos: {len(dataset_PSAX)}")
    
    # Generate samples from originals (before translation)
    print("\nCreating demo samples (original videos)...")
    
    num_samples = 5
    
    # A4C originals
    print("\nSaving A4C originals...")
    for i in range(min(num_samples, len(dataset_A4C))):
        video = dataset_A4C[i]
        save_video(video, os.path.join(output_dir, f"original_A4C_{i:03d}.mp4"))
    
    # PSAX originals
    print("\nSaving PSAX originals...")
    for i in range(min(num_samples, len(dataset_PSAX))):
        video = dataset_PSAX[i]
        save_video(video, os.path.join(output_dir, f"original_PSAX_{i:03d}.mp4"))
    
    # Check for trained models
    print("\nChecking for trained models...")
    checkpoint_AB = "checkpoints/G_AB_epoch10.pt"
    checkpoint_BA = "checkpoints/G_BA_epoch10.pt"
    
    if os.path.exists(checkpoint_AB) and os.path.exists(checkpoint_BA):
        print("  ✓ Found trained models! Generating translations...")
        
        from train_view_translation import VideoGenerator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        G_AB = VideoGenerator(1, 16, 64).to(device)
        G_BA = VideoGenerator(1, 16, 64).to(device)
        
        G_AB.load_state_dict(torch.load(checkpoint_AB, map_location=device))
        G_BA.load_state_dict(torch.load(checkpoint_BA, map_location=device))
        G_AB.eval()
        G_BA.eval()
        
        # Generate A4C -> PSAX
        print("\nGenerating A4C -> PSAX translations...")
        for i in range(min(num_samples, len(dataset_A4C))):
            real_A = dataset_A4C[i].unsqueeze(0).to(device)
            with torch.no_grad():
                fake_B = G_AB(real_A)
                rec_A = G_BA(fake_B)
            
            save_video(fake_B, os.path.join(output_dir, f"translated_A4C_to_PSAX_{i:03d}.mp4"))
            save_video(rec_A, os.path.join(output_dir, f"reconstructed_A4C_{i:03d}.mp4"))
        
        # Generate PSAX -> A4C
        print("\nGenerating PSAX -> A4C translations...")
        for i in range(min(num_samples, len(dataset_PSAX))):
            real_B = dataset_PSAX[i].unsqueeze(0).to(device)
            with torch.no_grad():
                fake_A = G_BA(real_B)
                rec_B = G_AB(fake_A)
            
            save_video(fake_A, os.path.join(output_dir, f"translated_PSAX_to_A4C_{i:03d}.mp4"))
            save_video(rec_B, os.path.join(output_dir, f"reconstructed_PSAX_{i:03d}.mp4"))
    else:
        print("  ⚠ No trained models found.")
        print("  To train models, run:")
        print("    python3 train_view_translation.py --epochs 10 --batch_size 2")
        print("  Then run this script again to generate translations.")
    
    print("\n" + "="*60)
    print(f"Demo complete! Videos saved in: {output_dir}/")
    print("="*60)
    
    # List generated files
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.mp4')])
    print(f"\nGenerated {len(files)} videos:")
    for f in files[:10]:
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files)-10} more")


if __name__ == "__main__":
    create_demo_samples()





