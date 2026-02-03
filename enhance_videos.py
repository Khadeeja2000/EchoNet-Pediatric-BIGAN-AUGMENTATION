"""
Enhance generated videos with super-resolution, temporal smoothing, and contrast enhancement
Upscales 64×64 → 128×128 and applies post-processing improvements
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def upscale_frame_simple(frame, target_size=128):
    """Simple bicubic upscaling - no training needed"""
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)


def enhance_contrast(frame):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)


def temporal_smoothing(video, sigma=1.0):
    """Apply Gaussian temporal smoothing to reduce flickering"""
    return gaussian_filter1d(video, sigma=sigma, axis=0)


def enhance_video(video_path, output_path, apply_super_resolution=True, 
                  apply_temporal_smoothing=True, apply_contrast=True, target_size=128):
    """
    Enhance a single video with multiple techniques
    
    Args:
        video_path: Path to input video (.npy file)
        output_path: Path to save enhanced video
        apply_super_resolution: Upscale 64→128
        apply_temporal_smoothing: Reduce flickering
        apply_contrast: Improve contrast
        target_size: Target resolution (default 128)
    """
    # Load video
    if video_path.endswith('.npy'):
        video = np.load(video_path)  # [T, H, W]
    else:
        # Try to load as MP4
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
    
    original_shape = video.shape
    print(f"  Original shape: {original_shape}")
    
    # Ensure video is in [T, H, W] format
    if len(video.shape) == 4:
        video = video.squeeze()
    if len(video.shape) == 5:
        video = video.squeeze()
    
    # Get current resolution
    T, H, W = video.shape
    
    # Step 1: Super-resolution (64→128)
    if apply_super_resolution and H < target_size:
        print(f"  Upscaling: {H}×{W} → {target_size}×{target_size}")
        enhanced_frames = []
        for frame in video:
            upscaled_frame = upscale_frame_simple(frame, target_size=target_size)
            enhanced_frames.append(upscaled_frame)
        video = np.array(enhanced_frames)
        print(f"  After upscaling: {video.shape}")
    else:
        print(f"  Resolution already {H}×{W}, skipping upscaling")
    
    # Step 2: Temporal smoothing
    if apply_temporal_smoothing:
        print(f"  Applying temporal smoothing...")
        video = temporal_smoothing(video, sigma=1.0)
        print(f"  After smoothing: {video.shape}")
    
    # Step 3: Contrast enhancement
    if apply_contrast:
        print(f"  Enhancing contrast...")
        enhanced_frames = []
        for frame in video:
            enhanced_frame = enhance_contrast(frame)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
        print(f"  After contrast: {video.shape}")
    
    # Save enhanced video
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy array (most reliable)
    np.save(output_path.replace('.mp4', '.npy'), video)
    
    # Also try to save as MP4
    try:
        import imageio
        # Convert to uint8
        video_uint8 = video.astype(np.uint8)
        # Convert to list of frames
        frames = [video_uint8[t] for t in range(video_uint8.shape[0])]
        # Save with imageio
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', pixelformat='gray')
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ⚠ Could not save MP4 (saved .npy instead): {e}")
    
    return video


def enhance_directory(input_dir, output_dir, pattern="*.npy", **kwargs):
    """Enhance all videos in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    if pattern.endswith('.npy'):
        video_files = list(input_path.glob(pattern))
        video_files.extend(list(input_path.glob("*.mp4")))
    else:
        video_files = list(input_path.glob(pattern))
    
    print(f"\n{'='*60}")
    print(f"Enhancing Videos")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(video_files)} videos")
    print(f"{'='*60}\n")
    
    if len(video_files) == 0:
        print("⚠ No video files found!")
        return
    
    enhanced_count = 0
    for video_file in tqdm(video_files, desc="Enhancing"):
        try:
            # Create output filename
            output_filename = f"enhanced_{video_file.name}"
            output_filepath = output_path / output_filename
            
            print(f"\nProcessing: {video_file.name}")
            
            # Enhance video
            enhanced_video = enhance_video(
                str(video_file),
                str(output_filepath),
                **kwargs
            )
            
            enhanced_count += 1
            print(f"  ✓ Enhanced: {video_file.name} → {output_filename}")
            
        except Exception as e:
            print(f"  ✗ Error processing {video_file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Enhancement Complete!")
    print(f"{'='*60}")
    print(f"Enhanced {enhanced_count}/{len(video_files)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance generated videos with super-resolution and post-processing"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="final_videos",
        help="Directory containing generated videos"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="final_videos_enhanced",
        help="Output directory for enhanced videos"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        default=128,
        help="Target resolution (default: 128, upscales from 64)"
    )
    
    parser.add_argument(
        "--no_super_resolution",
        action="store_true",
        help="Skip super-resolution upscaling"
    )
    
    parser.add_argument(
        "--no_temporal_smoothing",
        action="store_true",
        help="Skip temporal smoothing"
    )
    
    parser.add_argument(
        "--no_contrast",
        action="store_true",
        help="Skip contrast enhancement"
    )
    
    parser.add_argument(
        "--single_file",
        type=str,
        default=None,
        help="Enhance a single file instead of directory"
    )
    
    args = parser.parse_args()
    
    # Set enhancement options
    enhance_options = {
        "apply_super_resolution": not args.no_super_resolution,
        "apply_temporal_smoothing": not args.no_temporal_smoothing,
        "apply_contrast": not args.no_contrast,
        "target_size": args.target_size
    }
    
    print("\n" + "="*60)
    print("Video Enhancement Tool")
    print("="*60)
    print(f"Super-resolution: {'✓' if enhance_options['apply_super_resolution'] else '✗'}")
    print(f"Temporal smoothing: {'✓' if enhance_options['apply_temporal_smoothing'] else '✗'}")
    print(f"Contrast enhancement: {'✓' if enhance_options['apply_contrast'] else '✗'}")
    print(f"Target resolution: {args.target_size}×{args.target_size}")
    print("="*60 + "\n")
    
    if args.single_file:
        # Enhance single file
        output_file = args.output_dir + "/enhanced_" + os.path.basename(args.single_file)
        os.makedirs(args.output_dir, exist_ok=True)
        enhance_video(args.single_file, output_file, **enhance_options)
    else:
        # Enhance directory
        enhance_directory(args.input_dir, args.output_dir, **enhance_options)


if __name__ == "__main__":
    main()







