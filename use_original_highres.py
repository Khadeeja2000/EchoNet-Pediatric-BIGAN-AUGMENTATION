"""
Use ORIGINAL high-resolution videos instead of enhancing low-res GAN outputs
This is the REAL solution - use the source videos at their native resolution
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def load_original_video(video_path, target_frames=32, target_size=None):
    """
    Load original video and optionally resize
    target_size=None means keep original resolution
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize if target_size specified, otherwise keep original
            if target_size is not None:
                frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
    else:
        # Fallback: read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if target_size is not None:
                frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if len(frames) >= target_frames:
                break
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Pad or trim to target_frames
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8))
    
    frames = frames[:target_frames]
    
    video = np.array(frames)
    return video


def process_original_videos(manifest_path, output_dir, target_frames=32, target_size=None):
    """
    Process original videos from manifest
    target_size=None = keep original high resolution
    """
    print("\n" + "="*60)
    print("Using ORIGINAL High-Resolution Videos")
    print("="*60)
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    if target_size:
        print(f"Resizing to: {target_size}×{target_size}")
    else:
        print("Keeping ORIGINAL resolution (no downscaling)")
    print(f"Frames: {target_frames}")
    print("="*60)
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use original file_path (not processed_path)
    video_path_col = 'file_path' if 'file_path' in df.columns else 'processed_path'
    
    processed_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        original_video_path = row[video_path_col]
        
        if not os.path.exists(original_video_path):
            continue
        
        try:
            # Load original video
            video = load_original_video(original_video_path, target_frames=target_frames, target_size=target_size)
            
            if video is None:
                continue
            
            # Save
            base_name = Path(original_video_path).stem
            output_file = output_path / f"original_{base_name}.npy"
            output_mp4 = output_path / f"original_{base_name}.mp4"
            
            np.save(output_file, video)
            
            # Save as MP4
            try:
                import imageio
                frames = [video[t] for t in range(video.shape[0])]
                imageio.mimsave(str(output_mp4), frames, fps=30, codec='libx264', pixelformat='gray')
            except Exception:
                pass
            
            processed_count += 1
            
            if idx < 3:  # Print info for first 3
                print(f"\n  Sample {idx+1}: {base_name}")
                print(f"    Resolution: {video.shape[1]}×{video.shape[2]}")
                print(f"    Frames: {video.shape[0]}")
                print(f"    Range: [{video.min()}, {video.max()}]")
        
        except Exception as e:
            print(f"\n  Error processing {original_video_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Processed {processed_count}/{len(df)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Use original high-resolution videos instead of enhancing low-res GAN outputs"
    )
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv",
                       help="Path to manifest CSV")
    parser.add_argument("--output_dir", type=str, default="final_videos_original",
                       help="Output directory")
    parser.add_argument("--target_frames", type=int, default=32,
                       help="Number of frames to extract")
    parser.add_argument("--target_size", type=int, default=None,
                       help="Target resolution (None = keep original high resolution)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Process only first N samples")
    
    args = parser.parse_args()
    
    # Load manifest and limit if needed
    if args.num_samples:
        df = pd.read_csv(args.manifest)
        df = df.head(args.num_samples)
        temp_manifest = Path(args.manifest).parent / "temp_manifest.csv"
        df.to_csv(temp_manifest, index=False)
        manifest_path = temp_manifest
    else:
        manifest_path = args.manifest
    
    process_original_videos(
        manifest_path,
        args.output_dir,
        target_frames=args.target_frames,
        target_size=args.target_size
    )
    
    # Cleanup temp manifest
    if args.num_samples and temp_manifest.exists():
        temp_manifest.unlink()


if __name__ == "__main__":
    main()





