"""
Generate videos using trained diffusion model
"""
import argparse
import os
import numpy as np
import torch
import cv2
from train_video_diffusion import VideoDiffusionModel, sample_diffusion


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
    """Generate video samples using diffusion"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = VideoDiffusionModel(cfg["channels"], cfg["frames"], cfg["size"]).to(device)
    
    if os.path.exists(cfg["checkpoint"]):
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
        print(f"✓ Loaded: {cfg['checkpoint']}")
    else:
        print(f"⚠ Checkpoint not found: {cfg['checkpoint']}")
        return
    
    model.eval()
    
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    print(f"\nGenerating {cfg['num_samples']} videos...")
    for i in range(cfg["num_samples"]):
        print(f"  Generating video {i+1}/{cfg['num_samples']}...")
        
        # Generate video
        shape = (1, cfg["channels"], cfg["frames"], cfg["size"], cfg["size"])
        generated_video = sample_diffusion(model, shape, device, timesteps=cfg["timesteps"])
        
        # Save
        output_path = os.path.join(cfg["output_dir"], f"diffusion_generated_{i:03d}.mp4")
        save_video(generated_video[0], output_path)
    
    print(f"\n✓ Generated {cfg['num_samples']} videos in {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/diffusion_epoch10.pt")
    parser.add_argument("--output_dir", type=str, default="diffusion_generated")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100)
    args = parser.parse_args()
    
    cfg = {
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "frames": args.frames,
        "size": args.size,
        "channels": args.channels,
        "timesteps": args.timesteps,
    }
    
    generate_samples(cfg)




