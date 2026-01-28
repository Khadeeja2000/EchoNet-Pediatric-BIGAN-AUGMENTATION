"""
Compare all enhancement methods side-by-side to find the best one
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
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


def create_comparison_all_methods(video_name, base_dir, output_path):
    """Create comparison of all enhancement methods"""
    print(f"\nCreating comparison for: {video_name}")
    
    methods = {
        'Original': f"{base_dir}/final_videos/{video_name}",
        'Simple Cubic': f"{base_dir}/final_videos_quality/quality_{video_name.replace('.npy', '')}.npy",
        'Iterative': f"{base_dir}/final_videos_nn/nn_{video_name.replace('.npy', '')}.npy",
        'ESRGAN (old)': f"{base_dir}/final_videos_esrgan/esrgan_{video_name.replace('.npy', '')}.npy",
        'Improved': f"{base_dir}/final_videos_enhanced_improved/enhanced_{video_name.replace('.npy', '')}.npy",
    }
    
    # Load all videos
    videos = {}
    for method_name, path in methods.items():
        path_obj = Path(path)
        if path_obj.exists():
            try:
                videos[method_name] = load_video(str(path))
                print(f"  ✓ Loaded {method_name}: {videos[method_name].shape}")
            except Exception as e:
                print(f"  ✗ Failed to load {method_name}: {e}")
        else:
            print(f"  ⚠ Not found: {path}")
    
    if len(videos) < 2:
        print("  Not enough videos to compare")
        return
    
    # Get frame to compare (middle frame)
    frame_idx = len(list(videos.values())[0]) // 2
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    methods_list = list(videos.keys())
    for i, method_name in enumerate(methods_list[:6]):
        row = i // 3
        col = i % 3
        
        video = videos[method_name]
        frame = video[frame_idx]
        
        # Resize to same size for comparison
        if frame.shape[0] < 128:
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_NEAREST)
        elif frame.shape[0] > 128:
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{method_name}\n{video.shape}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'All Enhancement Methods Comparison - Frame {frame_idx+1}\n{video_name}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all enhancement methods")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory containing all video folders")
    parser.add_argument("--output_dir", type=str, default="comparisons_all_methods",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of videos to compare")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find original videos
    original_dir = base_dir / "final_videos"
    video_files = list(original_dir.glob("*.npy"))[:args.num_samples]
    
    print(f"\n{'='*60}")
    print(f"Comparing All Enhancement Methods")
    print(f"{'='*60}")
    print(f"Found {len(video_files)} videos to compare")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    for video_file in video_files:
        video_name = video_file.name
        output_path = output_dir / f"comparison_all_{video_name.replace('.npy', '')}.png"
        create_comparison_all_methods(video_name, str(base_dir), str(output_path))
    
    print(f"\n{'='*60}")
    print(f"Comparison Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()





