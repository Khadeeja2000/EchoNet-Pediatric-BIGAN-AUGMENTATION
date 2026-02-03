"""
High-Quality Video Enhancement - Simple and Effective Approach
Focuses on preserving quality while improving resolution
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_video(video_path):
    """Load video from .npy or .mp4 file"""
    if video_path.endswith('.npy'):
        video = np.load(video_path)
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        video = np.array(frames)
    
    if len(video.shape) == 4:
        video = video.squeeze()
    if len(video.shape) == 5:
        video = video.squeeze()
    
    if video.dtype != np.uint8:
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = np.clip(video, 0, 255).astype(np.uint8)
    
    return video


def upscale_simple_high_quality(frame, scale=2):
    """
    Simple high-quality upscaling - just use the best interpolation
    No aggressive processing that can degrade quality
    """
    # Use cubic interpolation - smooth and high quality
    upscaled = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), 
                         interpolation=cv2.INTER_CUBIC)
    return upscaled


def upscale_lanczos_clean(frame, scale=2):
    """
    Clean Lanczos upscaling - no post-processing
    """
    upscaled = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), 
                         interpolation=cv2.INTER_LANCZOS4)
    return upscaled


def enhance_video_simple(video_path, output_path, method='cubic', target_size=128):
    """
    Simple enhancement - just upscale with high-quality interpolation
    No aggressive processing that can degrade quality
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Load video
    video = load_video(video_path)
    T, H, W = video.shape
    print(f"  Original: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]")
    
    # Simple upscaling only
    if H < target_size:
        print(f"  Upscaling: {H}×{W} → {target_size}×{target_size} (method: {method})")
        enhanced_frames = []
        for frame in tqdm(video, desc="    Upscaling", leave=False):
            if method == 'cubic':
                upscaled = upscale_simple_high_quality(frame, scale=2)
            elif method == 'lanczos':
                upscaled = upscale_lanczos_clean(frame, scale=2)
            else:
                upscaled = upscale_simple_high_quality(frame, scale=2)
            
            # Resize to exact target if needed
            if upscaled.shape[0] != target_size:
                upscaled = cv2.resize(upscaled, (target_size, target_size), 
                                     interpolation=cv2.INTER_CUBIC)
            
            enhanced_frames.append(upscaled)
        video = np.array(enhanced_frames)
    else:
        print(f"  Resolution already {H}×{W}")
    
    print(f"  Final: {video.shape}, range: [{video.min()}, {video.max()}]")
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(output_path.replace('.mp4', '.npy'), video)
    
    try:
        import imageio
        frames = [video[t] for t in range(video.shape[0])]
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', pixelformat='gray')
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ⚠ Saved .npy only: {e}")
    
    return video


def enhance_directory_simple(input_dir, output_dir, method='cubic', num_samples=None, target_size=128):
    """Enhance videos with simple high-quality method"""
    print("\n" + "="*60)
    print("Simple High-Quality Video Enhancement")
    print("="*60)
    print(f"Method: {method} interpolation (no aggressive processing)")
    print(f"Target resolution: {target_size}×{target_size}")
    print("="*60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    video_files = [f for f in video_files if 'enhanced' not in f.name and 'esrgan' not in f.name]
    
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"\nFound {len(video_files)} videos to enhance\n")
    
    enhanced_count = 0
    for video_file in video_files:
        try:
            output_filename = f"quality_{video_file.stem}.mp4"
            output_filepath = output_path / output_filename
            
            enhance_video_simple(str(video_file), str(output_filepath), method=method, target_size=target_size)
            enhanced_count += 1
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    print(f"{'='*60}")
    print(f"Enhanced {enhanced_count}/{len(video_files)} videos")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Simple high-quality video enhancement")
    parser.add_argument("--input_dir", type=str, default="final_videos")
    parser.add_argument("--output_dir", type=str, default="final_videos_quality")
    parser.add_argument("--method", type=str, default="cubic", choices=['cubic', 'lanczos'])
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    enhance_directory_simple(
        args.input_dir,
        args.output_dir,
        method=args.method,
        num_samples=args.num_samples,
        target_size=args.target_size
    )


if __name__ == "__main__":
    main()





