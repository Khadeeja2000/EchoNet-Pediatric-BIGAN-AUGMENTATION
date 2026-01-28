"""
Improved Real-ESRGAN Video Enhancement for Echocardiogram Videos
Uses better algorithms and more conservative settings for superior quality
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
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
    
    # Ensure video is in [T, H, W] format
    if len(video.shape) == 4:
        video = video.squeeze()
    if len(video.shape) == 5:
        video = video.squeeze()
    
    # Normalize to uint8 if needed
    if video.dtype != np.uint8:
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = np.clip(video, 0, 255).astype(np.uint8)
    
    return video


def upscale_frame_edsr_style(frame, scale=2):
    """
    High-quality upscaling using EDSR-style approach
    Uses iterative refinement and edge-preserving techniques
    """
    # Step 1: Upscale with cubic interpolation (smooth base)
    upscaled = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply edge-preserving bilateral filter (preserves edges, smooths noise)
    filtered = cv2.bilateralFilter(upscaled, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Step 3: Subtle sharpening using unsharp mask (much more conservative)
    gaussian = cv2.GaussianBlur(filtered, (0, 0), 1.5)
    sharpened = cv2.addWeighted(filtered, 1.2, gaussian, -0.2, 0)  # Much lighter sharpening
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def upscale_frame_iterative(frame, scale=2):
    """
    Iterative upscaling - upscale in steps for better quality
    """
    current = frame.astype(np.float32)
    
    # Upscale in steps (better than direct 2x)
    for step in range(scale):
        h, w = current.shape
        # Use cubic for smooth upscaling
        current = cv2.resize(current, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        # Light denoising
        current = cv2.bilateralFilter(current.astype(np.uint8), d=3, sigmaColor=30, sigmaSpace=30).astype(np.float32)
    
    # Final light sharpening
    gaussian = cv2.GaussianBlur(current.astype(np.uint8), (0, 0), 1.0)
    sharpened = cv2.addWeighted(current.astype(np.uint8), 1.1, gaussian, -0.1, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_frame_esrgan(frame, upscaler, dnn_sr=None, method='edsr'):
    """Enhance a single frame using Real-ESRGAN or high-quality fallback"""
    if upscaler is not None:
        # Real-ESRGAN expects RGB input
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame
        
        try:
            enhanced_frame, _ = upscaler.enhance(frame_rgb, outscale=2)
            if len(enhanced_frame.shape) == 3:
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2GRAY)
            return enhanced_frame
        except Exception:
            pass
    
    # High-quality fallback methods
    if method == 'edsr':
        return upscale_frame_edsr_style(frame, scale=2)
    elif method == 'iterative':
        return upscale_frame_iterative(frame, scale=2)
    else:
        # Conservative Lanczos
        upscaled = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), 
                             interpolation=cv2.INTER_LANCZOS4)
        # Very light sharpening
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.05, gaussian, -0.05, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast_adaptive(frame, clip_limit=2.0):
    """Adaptive contrast enhancement with better parameters"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(frame)


def temporal_smoothing_adaptive(video, sigma=0.3):
    """Very light temporal smoothing to preserve detail"""
    smoothed = gaussian_filter1d(video.astype(float), sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def denoise_frame(frame):
    """Light denoising to reduce artifacts"""
    # Use non-local means denoising (preserves edges better)
    denoised = cv2.fastNlMeansDenoising(frame, h=5, templateWindowSize=7, searchWindowSize=21)
    return denoised


def enhance_video_improved(video_path, output_path, upscaler=None, 
                          apply_temporal_smoothing=True, 
                          apply_contrast=True,
                          apply_denoising=True,
                          target_size=128,
                          dnn_sr=None,
                          method='edsr'):
    """
    Improved video enhancement with better algorithms
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Load video
    video = load_video(video_path)
    T, H, W = video.shape
    print(f"  Original shape: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]")
    
    # Step 0: Optional light denoising (before upscaling)
    if apply_denoising:
        print(f"  Applying light denoising...")
        denoised_frames = []
        for frame in tqdm(video, desc="    Denoising", leave=False):
            denoised_frame = denoise_frame(frame)
            denoised_frames.append(denoised_frame)
        video = np.array(denoised_frames)
    
    # Step 1: High-quality super-resolution upscaling
    if H < target_size:
        print(f"  Upscaling: {H}×{W} → {H*2}×{W*2} (method: {method})")
        enhanced_frames = []
        for i, frame in enumerate(tqdm(video, desc="    Upscaling frames", leave=False)):
            enhanced_frame = enhance_frame_esrgan(frame, upscaler, dnn_sr, method=method)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
        print(f"  After upscaling: {video.shape}")
        
        # Resize to target if needed
        if target_size != H*2:
            print(f"  Resizing to target: {target_size}×{target_size}")
            resized_frames = []
            for frame in video:
                resized_frame = cv2.resize(frame, (target_size, target_size), 
                                         interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized_frame)
            video = np.array(resized_frames)
    else:
        print(f"  Resolution already {H}×{W}, skipping upscaling")
    
    # Step 2: Very light temporal smoothing (preserves detail)
    if apply_temporal_smoothing:
        print(f"  Applying light temporal smoothing...")
        video = temporal_smoothing_adaptive(video, sigma=0.3)
    
    # Step 3: Adaptive contrast enhancement
    if apply_contrast:
        print(f"  Enhancing contrast (adaptive)...")
        enhanced_frames = []
        for frame in video:
            enhanced_frame = enhance_contrast_adaptive(frame, clip_limit=2.0)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
    
    print(f"  Final shape: {video.shape}, range: [{video.min()}, {video.max()}]")
    
    # Save enhanced video
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy array
    np.save(output_path.replace('.mp4', '.npy'), video)
    
    # Save as MP4
    try:
        import imageio
        video_uint8 = video.astype(np.uint8)
        frames = [video_uint8[t] for t in range(video_uint8.shape[0])]
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', pixelformat='gray')
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ⚠ Saved .npy only (MP4 failed): {e}")
    
    return video


def enhance_directory_improved(input_dir, output_dir, model_name='realesrgan-x2plus',
                              num_samples=None, method='edsr', **kwargs):
    """Enhance videos with improved methods"""
    print("\n" + "="*60)
    print("Improved Video Enhancement")
    print("="*60)
    
    upscaler = None
    dnn_sr = None
    
    # Try Real-ESRGAN first
    try:
        from realesrgan import RealESRGANer
        upscaler = RealESRGANer(
            scale=2,
            model_path=None,
            model=model_name,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        print(f"✓ Real-ESRGAN initialized with model: {model_name}")
    except Exception as e:
        print(f"Using high-quality fallback method: {method}")
        upscaler = None
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    video_files = [f for f in video_files if 'enhanced' not in f.name and 'esrgan' not in f.name]
    
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"\n{'='*60}")
    print(f"Enhanced Video Processing")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    print(f"Found {len(video_files)} videos to enhance")
    print(f"{'='*60}\n")
    
    if len(video_files) == 0:
        print("⚠ No video files found!")
        return
    
    enhanced_count = 0
    for video_file in video_files:
        try:
            output_filename = f"enhanced_{video_file.stem}.mp4"
            output_filepath = output_path / output_filename
            
            enhanced_video = enhance_video_improved(
                str(video_file),
                str(output_filepath),
                upscaler,
                dnn_sr=dnn_sr,
                method=method,
                **kwargs
            )
            
            enhanced_count += 1
            print(f"  ✓ Enhanced: {video_file.name} → {output_filename}\n")
            
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
        description="Improved video enhancement with better algorithms"
    )
    
    parser.add_argument("--input_dir", type=str, default="final_videos",
                       help="Directory containing original videos")
    parser.add_argument("--output_dir", type=str, default="final_videos_enhanced_improved",
                       help="Output directory for enhanced videos")
    parser.add_argument("--model", type=str, default="realesrgan-x2plus",
                       choices=['realesrgan-x2plus', 'realesrgan-x4plus'],
                       help="Real-ESRGAN model")
    parser.add_argument("--method", type=str, default="edsr",
                       choices=['edsr', 'iterative', 'lanczos'],
                       help="Fallback upscaling method")
    parser.add_argument("--target_size", type=int, default=128,
                       help="Target resolution")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of videos to enhance")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Skip temporal smoothing")
    parser.add_argument("--no_contrast", action="store_true",
                       help="Skip contrast enhancement")
    parser.add_argument("--no_denoising", action="store_true",
                       help="Skip denoising")
    
    args = parser.parse_args()
    
    enhance_options = {
        "apply_temporal_smoothing": not args.no_temporal,
        "apply_contrast": not args.no_contrast,
        "apply_denoising": not args.no_denoising,
        "target_size": args.target_size,
        "model_name": args.model,
        "num_samples": args.num_samples,
        "method": args.method
    }
    
    print("\n" + "="*60)
    print("Improved Video Enhancement Tool")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Denoising: {'✓' if enhance_options['apply_denoising'] else '✗'}")
    print(f"Temporal smoothing: {'✓' if enhance_options['apply_temporal_smoothing'] else '✗'}")
    print(f"Contrast enhancement: {'✓' if enhance_options['apply_contrast'] else '✗'}")
    print(f"Target resolution: {args.target_size}×{args.target_size}")
    print("="*60)
    
    enhance_directory_improved(
        args.input_dir,
        args.output_dir,
        **enhance_options
    )


if __name__ == "__main__":
    main()





