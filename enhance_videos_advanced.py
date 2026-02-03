"""
Advanced video enhancement with better super-resolution methods
Uses sharper interpolation and additional sharpening techniques
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage


def upscale_frame_lanczos(frame, target_size=128):
    """Lanczos interpolation - sharper than bicubic"""
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def upscale_frame_sharp(frame, target_size=128):
    """Upscale with sharpening"""
    # First upscale
    upscaled = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def upscale_frame_edge_preserving(frame, target_size=128):
    """Edge-preserving upscaling with sharpening"""
    # Upscale with Lanczos (sharper)
    upscaled = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(upscaled, 5, 50, 50)
    
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast_adaptive(frame):
    """Adaptive contrast enhancement"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(frame)


def sharpen_frame(frame, strength=1.5):
    """Apply unsharp masking"""
    gaussian = cv2.GaussianBlur(frame, (0, 0), 1.0)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_video_advanced(video_path, output_path, method='sharp', 
                           apply_temporal_smoothing=True, apply_contrast=True, 
                           apply_sharpening=True, target_size=128):
    """
    Advanced video enhancement with better super-resolution
    
    Methods:
    - 'lanczos': Lanczos interpolation (sharper than bicubic)
    - 'sharp': Upscale + unsharp masking
    - 'edge_preserving': Edge-preserving upscaling with sharpening
    """
    # Load video
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
    
    # Ensure video is in [T, H, W] format
    if len(video.shape) == 4:
        video = video.squeeze()
    if len(video.shape) == 5:
        video = video.squeeze()
    
    T, H, W = video.shape
    
    # Ensure uint8 format
    if video.dtype != np.uint8:
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)
    
    print(f"  Original: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]")
    
    # Step 1: Super-resolution with better method
    if H < target_size:
        print(f"  Upscaling: {H}×{W} → {target_size}×{target_size} (method: {method})")
        enhanced_frames = []
        for i, frame in enumerate(video):
            if method == 'lanczos':
                upscaled_frame = upscale_frame_lanczos(frame, target_size)
            elif method == 'sharp':
                upscaled_frame = upscale_frame_sharp(frame, target_size)
            elif method == 'edge_preserving':
                upscaled_frame = upscale_frame_edge_preserving(frame, target_size)
            else:
                upscaled_frame = upscale_frame_sharp(frame, target_size)
            
            enhanced_frames.append(upscaled_frame)
        video = np.array(enhanced_frames)
        print(f"  After upscaling: {video.shape}")
    
    # Step 2: Temporal smoothing (lighter to preserve detail)
    if apply_temporal_smoothing:
        print(f"  Applying light temporal smoothing...")
        video = gaussian_filter1d(video.astype(float), sigma=0.5, axis=0)
        video = video.astype(np.uint8)
    
    # Step 3: Contrast enhancement
    if apply_contrast:
        print(f"  Enhancing contrast...")
        enhanced_frames = []
        for frame in video:
            enhanced_frame = enhance_contrast_adaptive(frame)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
    
    # Step 4: Additional sharpening
    if apply_sharpening:
        print(f"  Applying sharpening...")
        enhanced_frames = []
        for frame in video:
            sharpened_frame = sharpen_frame(frame, strength=1.2)
            enhanced_frames.append(sharpened_frame)
        video = np.array(enhanced_frames)
    
    # Save enhanced video
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy array
    np.save(output_path.replace('.mp4', '.npy'), video)
    
    # Try to save as MP4
    try:
        import imageio
        video_uint8 = video.astype(np.uint8)
        frames = [video_uint8[t] for t in range(video_uint8.shape[0])]
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', pixelformat='gray')
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ⚠ Saved .npy only: {e}")
    
    return video


def enhance_directory_advanced(input_dir, output_dir, method='sharp', **kwargs):
    """Enhance all videos in a directory with advanced methods"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    
    print(f"\n{'='*60}")
    print(f"Advanced Video Enhancement")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    print(f"Found {len(video_files)} videos")
    print(f"{'='*60}\n")
    
    if len(video_files) == 0:
        print("⚠ No video files found!")
        return
    
    enhanced_count = 0
    for video_file in tqdm(video_files, desc="Enhancing"):
        try:
            # Skip already enhanced files
            if 'enhanced' in video_file.name:
                continue
                
            output_filename = f"enhanced_{video_file.name}"
            output_filepath = output_path / output_filename
            
            # Enhance video
            enhanced_video = enhance_video_advanced(
                str(video_file),
                str(output_filepath),
                method=method,
                **kwargs
            )
            
            enhanced_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {video_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Enhancement Complete!")
    print(f"{'='*60}")
    print(f"Enhanced {enhanced_count}/{len(video_files)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced video enhancement with better super-resolution"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="final_videos",
        help="Directory containing original videos"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="final_videos_enhanced_sharp",
        help="Output directory for enhanced videos"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="sharp",
        choices=['lanczos', 'sharp', 'edge_preserving'],
        help="Upscaling method: lanczos (sharper interpolation), sharp (with unsharp masking), edge_preserving (best quality)"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        default=128,
        help="Target resolution"
    )
    
    parser.add_argument(
        "--no_temporal",
        action="store_true",
        help="Skip temporal smoothing"
    )
    
    parser.add_argument(
        "--no_contrast",
        action="store_true",
        help="Skip contrast enhancement"
    )
    
    parser.add_argument(
        "--no_sharpening",
        action="store_true",
        help="Skip additional sharpening"
    )
    
    args = parser.parse_args()
    
    enhance_options = {
        "apply_temporal_smoothing": not args.no_temporal,
        "apply_contrast": not args.no_contrast,
        "apply_sharpening": not args.no_sharpening,
        "target_size": args.target_size
    }
    
    print("\n" + "="*60)
    print("Advanced Video Enhancement Tool")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Temporal smoothing: {'✓' if enhance_options['apply_temporal_smoothing'] else '✗'}")
    print(f"Contrast enhancement: {'✓' if enhance_options['apply_contrast'] else '✗'}")
    print(f"Sharpening: {'✓' if enhance_options['apply_sharpening'] else '✗'}")
    print(f"Target resolution: {args.target_size}×{args.target_size}")
    print("="*60 + "\n")
    
    enhance_directory_advanced(
        args.input_dir,
        args.output_dir,
        method=args.method,
        **enhance_options
    )


if __name__ == "__main__":
    main()







