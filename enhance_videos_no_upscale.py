"""
Enhance video quality WITHOUT upscaling - improve at original resolution
Sometimes less is more - focus on quality improvement, not resolution increase
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


def enhance_frame_quality(frame):
    """
    Enhance frame quality at original resolution
    Focus on: denoising, contrast, sharpening - NO upscaling
    """
    # Step 1: Light denoising (preserves edges)
    denoised = cv2.fastNlMeansDenoising(frame, h=3, templateWindowSize=5, searchWindowSize=15)
    
    # Step 2: Adaptive contrast enhancement (subtle)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    contrast = clahe.apply(denoised)
    
    # Step 3: Very light sharpening (just to enhance edges)
    gaussian = cv2.GaussianBlur(contrast, (0, 0), 0.8)
    sharpened = cv2.addWeighted(contrast, 1.05, gaussian, -0.05, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_video_no_upscale(video_path, output_path):
    """
    Enhance video quality without changing resolution
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Load video
    video = load_video(video_path)
    T, H, W = video.shape
    print(f"  Original: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]")
    
    # Enhance each frame (no upscaling)
    print(f"  Enhancing quality at original resolution {H}×{W}...")
    enhanced_frames = []
    for frame in tqdm(video, desc="    Enhancing", leave=False):
        enhanced_frame = enhance_frame_quality(frame)
        enhanced_frames.append(enhanced_frame)
    video = np.array(enhanced_frames)
    
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


def enhance_directory_no_upscale(input_dir, output_dir, num_samples=None):
    """Enhance videos without upscaling"""
    print("\n" + "="*60)
    print("Quality Enhancement (NO Upscaling)")
    print("="*60)
    print("Improving: denoising, contrast, sharpness")
    print("Keeping: original resolution")
    print("="*60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    video_files = [f for f in video_files if 'enhanced' not in f.name and 'esrgan' not in f.name 
                   and 'quality' not in f.name and 'nn' not in f.name]
    
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"\nFound {len(video_files)} videos to enhance\n")
    
    enhanced_count = 0
    for video_file in video_files:
        try:
            output_filename = f"no_upscale_{video_file.stem}.mp4"
            output_filepath = output_path / output_filename
            
            enhance_video_no_upscale(str(video_file), str(output_filepath))
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
    parser = argparse.ArgumentParser(description="Enhance quality without upscaling")
    parser.add_argument("--input_dir", type=str, default="final_videos")
    parser.add_argument("--output_dir", type=str, default="final_videos_no_upscale")
    parser.add_argument("--num_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    enhance_directory_no_upscale(
        args.input_dir,
        args.output_dir,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()





