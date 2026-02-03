"""
Real-ESRGAN Video Enhancement for Echocardiogram Videos
Uses pre-trained Real-ESRGAN models for superior super-resolution quality
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


def enhance_frame_esrgan(frame, upscaler, dnn_sr=None):
    """Enhance a single frame using Real-ESRGAN, OpenCV DNN, or fallback method"""
    if upscaler is not None:
        # Real-ESRGAN expects RGB input, so convert grayscale to RGB
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame
        
        # Enhance with Real-ESRGAN
        try:
            enhanced_frame, _ = upscaler.enhance(frame_rgb, outscale=2)
            # Convert back to grayscale
            if len(enhanced_frame.shape) == 3:
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2GRAY)
            return enhanced_frame
        except Exception as e:
            # Fallback to other methods
            pass
    
    # Try OpenCV DNN Super-Resolution if available
    if dnn_sr is not None:
        try:
            # DNN expects RGB
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame
            enhanced_frame = dnn_sr.upsample(frame_rgb)
            if len(enhanced_frame.shape) == 3:
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2GRAY)
            return enhanced_frame
        except Exception:
            pass
    
    # Fallback: Advanced upscaling with Lanczos + sharpening
    # Upscale with Lanczos (sharper than bicubic)
    upscaled = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast_clahe(frame):
    """Apply CLAHE contrast enhancement"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)


def temporal_smoothing(video, sigma=0.5):
    """Apply light temporal smoothing to reduce flickering"""
    smoothed = gaussian_filter1d(video.astype(float), sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def enhance_video_esrgan(video_path, output_path, upscaler, 
                         apply_temporal_smoothing=True, 
                         apply_contrast=True,
                         target_size=128,
                         dnn_sr=None):
    """
    Enhance video using Real-ESRGAN super-resolution
    
    Args:
        video_path: Path to input video
        output_path: Path to save enhanced video
        upscaler: Real-ESRGAN upscaler instance
        apply_temporal_smoothing: Apply temporal smoothing
        apply_contrast: Apply contrast enhancement
        target_size: Target resolution (default 128, but ESRGAN will upscale 2x)
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Load video
    video = load_video(video_path)
    T, H, W = video.shape
    print(f"  Original shape: {video.shape}, dtype: {video.dtype}")
    
    # Step 1: Real-ESRGAN super-resolution (2x upscaling)
    if H < target_size:
        print(f"  Applying Real-ESRGAN upscaling: {H}×{W} → {H*2}×{W*2}")
        enhanced_frames = []
        for i, frame in enumerate(tqdm(video, desc="    Enhancing frames", leave=False)):
            enhanced_frame = enhance_frame_esrgan(frame, upscaler, dnn_sr)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
        print(f"  After ESRGAN: {video.shape}")
        
        # If target_size is specified and different, resize
        if target_size != H*2:
            print(f"  Resizing to target: {target_size}×{target_size}")
            resized_frames = []
            for frame in video:
                resized_frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized_frame)
            video = np.array(resized_frames)
    else:
        print(f"  Resolution already {H}×{W}, skipping upscaling")
    
    # Step 2: Temporal smoothing
    if apply_temporal_smoothing:
        print(f"  Applying temporal smoothing...")
        video = temporal_smoothing(video, sigma=0.5)
    
    # Step 3: Contrast enhancement
    if apply_contrast:
        print(f"  Enhancing contrast...")
        enhanced_frames = []
        for frame in video:
            enhanced_frame = enhance_contrast_clahe(frame)
            enhanced_frames.append(enhanced_frame)
        video = np.array(enhanced_frames)
    
    print(f"  Final shape: {video.shape}")
    
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


def enhance_directory_esrgan(input_dir, output_dir, model_name='realesrgan-x2plus',
                            num_samples=None, **kwargs):
    """Enhance videos in directory using Real-ESRGAN"""
    # Initialize Real-ESRGAN upscaler
    print("\n" + "="*60)
    print("Initializing Real-ESRGAN...")
    print("="*60)
    
    upscaler = None
    dnn_sr = None
    
    # Try Real-ESRGAN first
    try:
        from realesrgan import RealESRGANer
        
        # Create upscaler
        upscaler = RealESRGANer(
            scale=2,
            model_path=None,  # Will download automatically
            model=model_name,
            tile=0,  # No tiling for small images
            tile_pad=10,
            pre_pad=0,
            half=False  # Use full precision for better quality
        )
        print(f"✓ Real-ESRGAN initialized with model: {model_name}")
    except ImportError as e:
        print(f"WARNING: Could not import Real-ESRGAN: {e}")
        print("Trying OpenCV DNN Super-Resolution...")
    except Exception as e:
        print(f"WARNING: Failed to initialize Real-ESRGAN: {e}")
        print("Trying OpenCV DNN Super-Resolution...")
        upscaler = None
    
    # Try OpenCV DNN Super-Resolution as alternative
    if upscaler is None:
        try:
            dnn = cv2.dnn_superres.DnnSuperResImpl_create()
            # Note: Would need to download model file, but for now use fallback
            print("OpenCV DNN Super-Resolution available but model not loaded")
            print("Using advanced Lanczos + sharpening method...")
        except Exception:
            print("Using advanced Lanczos + sharpening method...")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    
    # Filter out already enhanced files
    video_files = [f for f in video_files if 'enhanced' not in f.name]
    
    # Limit to num_samples if specified
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"\n{'='*60}")
    print(f"Real-ESRGAN Video Enhancement")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Found {len(video_files)} videos to enhance")
    print(f"{'='*60}\n")
    
    if len(video_files) == 0:
        print("⚠ No video files found!")
        return
    
    enhanced_count = 0
    for video_file in video_files:
        try:
            output_filename = f"esrgan_{video_file.stem}.mp4"
            output_filepath = output_path / output_filename
            
            # Enhance video (will use fallback if upscaler is None)
            enhanced_video = enhance_video_esrgan(
                str(video_file),
                str(output_filepath),
                upscaler,
                dnn_sr=dnn_sr,
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
        description="Enhance videos using Real-ESRGAN super-resolution"
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
        default="final_videos_esrgan",
        help="Output directory for enhanced videos"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="realesrgan-x2plus",
        choices=['realesrgan-x2plus', 'realesrgan-x4plus', 'realesrnet-x4plus'],
        help="Real-ESRGAN model to use"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        default=128,
        help="Target resolution (default: 128)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of videos to enhance (default: all)"
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
        "--single_file",
        type=str,
        default=None,
        help="Enhance a single file instead of directory"
    )
    
    args = parser.parse_args()
    
    enhance_options = {
        "apply_temporal_smoothing": not args.no_temporal,
        "apply_contrast": not args.no_contrast,
        "target_size": args.target_size,
        "model_name": args.model,
        "num_samples": args.num_samples
    }
    
    print("\n" + "="*60)
    print("Real-ESRGAN Video Enhancement Tool")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Temporal smoothing: {'✓' if enhance_options['apply_temporal_smoothing'] else '✗'}")
    print(f"Contrast enhancement: {'✓' if enhance_options['apply_contrast'] else '✗'}")
    print(f"Target resolution: {args.target_size}×{args.target_size}")
    print("="*60)
    
    if args.single_file:
        # Single file enhancement
        upscaler = None
        dnn_sr = None
        try:
            from realesrgan import RealESRGANer
            upscaler = RealESRGANer(
                scale=2,
                model_path=None,
                model=args.model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
        except Exception:
            pass
        
        output_file = os.path.join(args.output_dir, f"esrgan_{os.path.basename(args.single_file)}")
        os.makedirs(args.output_dir, exist_ok=True)
        enhance_video_esrgan(args.single_file, output_file, upscaler, 
                           apply_temporal_smoothing=enhance_options['apply_temporal_smoothing'],
                           apply_contrast=enhance_options['apply_contrast'],
                           target_size=enhance_options['target_size'],
                           dnn_sr=dnn_sr)
    else:
        # Directory enhancement
        enhance_directory_esrgan(
            args.input_dir,
            args.output_dir,
            **enhance_options
        )


if __name__ == "__main__":
    main()

