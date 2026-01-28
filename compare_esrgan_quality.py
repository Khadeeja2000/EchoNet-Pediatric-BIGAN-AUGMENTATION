"""
Create side-by-side comparison visualization of original vs ESRGAN-enhanced videos
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


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


def create_comparison_frames(original_video, enhanced_video, num_frames=6):
    """Create side-by-side comparison frames"""
    T = min(len(original_video), len(enhanced_video))
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    comparison_frames = []
    for idx in frame_indices:
        orig_frame = original_video[idx]
        enh_frame = enhanced_video[idx]
        
        # Resize original to match enhanced if needed
        if orig_frame.shape != enh_frame.shape:
            orig_frame = cv2.resize(orig_frame, (enh_frame.shape[1], enh_frame.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side comparison
        comparison = np.hstack([orig_frame, enh_frame])
        comparison_frames.append(comparison)
    
    return comparison_frames, frame_indices


def create_static_comparison(original_path, enhanced_path, output_path, num_frames=6):
    """Create static side-by-side comparison image"""
    print(f"\nCreating comparison: {Path(original_path).name}")
    
    # Load videos
    original_video = load_video(original_path)
    enhanced_video = load_video(enhanced_path)
    
    print(f"  Original: {original_video.shape}")
    print(f"  Enhanced: {enhanced_video.shape}")
    
    # Create comparison frames
    comparison_frames, frame_indices = create_comparison_frames(original_video, enhanced_video, num_frames)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, (comp_frame, idx) in enumerate(zip(comparison_frames, frame_indices)):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.imshow(comp_frame, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Frame {idx+1}/{len(original_video)}\nOriginal (left) vs Enhanced (right)', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Add main title
    fig.suptitle(f'Quality Comparison: {Path(original_path).stem}\n'
                f'Original: {original_video.shape[1]}×{original_video.shape[2]} → '
                f'Enhanced: {enhanced_video.shape[1]}×{enhanced_video.shape[2]}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_animated_comparison(original_path, enhanced_path, output_path):
    """Create animated side-by-side comparison video"""
    print(f"\nCreating animated comparison: {Path(original_path).name}")
    
    # Load videos
    original_video = load_video(original_path)
    enhanced_video = load_video(enhanced_path)
    
    T = min(len(original_video), len(enhanced_video))
    
    # Resize original to match enhanced if needed
    if original_video.shape[1:] != enhanced_video.shape[1:]:
        resized_original = []
        for frame in original_video:
            resized = cv2.resize(frame, (enhanced_video.shape[2], enhanced_video.shape[1]), 
                               interpolation=cv2.INTER_NEAREST)
            resized_original.append(resized)
        original_video = np.array(resized_original)
    
    # Create side-by-side frames
    comparison_frames = []
    for t in range(T):
        comparison = np.hstack([original_video[t], enhanced_video[t]])
        comparison_frames.append(comparison)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title(f'Original (left) vs Enhanced (right) - {Path(original_path).stem}', 
                fontsize=12, fontweight='bold')
    
    im = ax.imshow(comparison_frames[0], cmap='gray', vmin=0, vmax=255, animated=True)
    
    def animate(frame):
        im.set_array(comparison_frames[frame])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=T, interval=100, blit=True, repeat=True)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', fps=10, bitrate=1800)
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comparison visualizations")
    parser.add_argument("--original_dir", type=str, default="final_videos",
                       help="Directory with original videos")
    parser.add_argument("--enhanced_dir", type=str, default="final_videos_esrgan",
                       help="Directory with enhanced videos")
    parser.add_argument("--output_dir", type=str, default="esrgan_comparisons",
                       help="Output directory for comparisons")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of videos to compare")
    parser.add_argument("--create_animation", action="store_true",
                       help="Create animated comparisons")
    
    args = parser.parse_args()
    
    original_dir = Path(args.original_dir)
    enhanced_dir = Path(args.enhanced_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching video files
    enhanced_files = list(enhanced_dir.glob("esrgan_*.npy"))
    enhanced_files.extend(list(enhanced_dir.glob("esrgan_*.mp4")))
    enhanced_files.extend(list(enhanced_dir.glob("enhanced_*.npy")))
    enhanced_files.extend(list(enhanced_dir.glob("enhanced_*.mp4")))
    enhanced_files.extend(list(enhanced_dir.glob("original_*.npy")))
    enhanced_files.extend(list(enhanced_dir.glob("original_*.mp4")))
    
    if len(enhanced_files) == 0:
        print("No enhanced videos found!")
        return
    
    # Limit to num_samples
    enhanced_files = enhanced_files[:args.num_samples]
    
    print(f"\n{'='*60}")
    print(f"Creating Quality Comparisons")
    print(f"{'='*60}")
    print(f"Found {len(enhanced_files)} enhanced videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    for enhanced_file in enhanced_files:
        # Find corresponding original file
        base_name = enhanced_file.stem.replace("esrgan_", "").replace("enhanced_", "").replace("original_", "")
        original_file = None
        
        # Try .npy first
        original_candidate = original_dir / f"{base_name}.npy"
        if original_candidate.exists():
            original_file = original_candidate
        else:
            # Try .mp4
            original_candidate = original_dir / f"{base_name}.mp4"
            if original_candidate.exists():
                original_file = original_candidate
        
        if original_file is None:
            print(f"  ⚠ Could not find original for {enhanced_file.name}")
            continue
        
        # Create static comparison
        output_static = output_dir / f"comparison_{base_name}.png"
        create_static_comparison(str(original_file), str(enhanced_file), str(output_static))
        
        # Create animated comparison if requested
        if args.create_animation:
            output_anim = output_dir / f"comparison_{base_name}.mp4"
            create_animated_comparison(str(original_file), str(enhanced_file), str(output_anim))
    
    print(f"\n{'='*60}")
    print(f"Comparison Complete!")
    print(f"{'='*60}")
    print(f"Created comparisons for {len(enhanced_files)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

