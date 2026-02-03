"""
Extract and display frames from BiGAN-generated samples
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def extract_frames_from_video(video_path, num_frames=4):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None
    
    # Extract evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    
    cap.release()
    return frames

def display_bigan_samples():
    """Display frames from BiGAN-generated samples"""
    
    # Sample directories
    sample_dirs = [
        "augmentation/samples",
        "augmentation/stable_samples", 
        "augmentation/samples_highq"
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('BiGAN Generated Samples - Sample Frames', fontsize=16, fontweight='bold')
    
    row = 0
    for sample_dir in sample_dirs:
        if not os.path.exists(sample_dir):
            print(f"Directory {sample_dir} not found, skipping...")
            continue
            
        # Get first video from each directory
        video_files = sorted(list(Path(sample_dir).glob("*.mp4")))[:1]
        
        if len(video_files) == 0:
            print(f"No videos found in {sample_dir}")
            continue
            
        video_path = video_files[0]
        print(f"\nProcessing: {video_path.name}")
        
        # Extract frames
        frames = extract_frames_from_video(video_path, num_frames=4)
        
        if frames is None or len(frames) == 0:
            print(f"  Failed to extract frames from {video_path.name}")
            continue
        
        # Display frames
        for col, frame in enumerate(frames[:4]):
            ax = axes[row, col]
            ax.imshow(frame, cmap='gray')
            ax.set_title(f'{Path(sample_dir).name}\nFrame {col+1}', fontsize=10)
            ax.axis('off')
        
        row += 1
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "bigan_samples_preview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved preview to: {output_path}")
    
    plt.show()

def list_all_samples():
    """List all BiGAN sample files"""
    print("=" * 60)
    print("BiGAN Generated Samples")
    print("=" * 60)
    
    sample_dirs = {
        "augmentation/samples": "Original BiGAN samples",
        "augmentation/stable_samples": "Stable BiGAN samples", 
        "augmentation/samples_highq": "High quality BiGAN samples"
    }
    
    for sample_dir, description in sample_dirs.items():
        if os.path.exists(sample_dir):
            video_files = sorted(list(Path(sample_dir).glob("*.mp4")))
            print(f"\n{description}:")
            print(f"  Location: {sample_dir}")
            print(f"  Count: {len(video_files)} videos")
            if len(video_files) > 0:
                print(f"  Examples: {video_files[0].name}, {video_files[-1].name}")
        else:
            print(f"\n{description}:")
            print(f"  Location: {sample_dir} (NOT FOUND)")

if __name__ == "__main__":
    print("BiGAN Sample Viewer")
    print("=" * 60)
    
    # List all samples
    list_all_samples()
    
    # Display sample frames
    print("\n" + "=" * 60)
    print("Extracting sample frames...")
    display_bigan_samples()
    
    print("\n" + "=" * 60)
    print("To view individual videos, you can:")
    print("  - Open the MP4 files directly in a video player")
    print("  - Use: open augmentation/samples/sample_s0_a0_1.mp4")
    print("  - Or use QuickTime/Media Player")







