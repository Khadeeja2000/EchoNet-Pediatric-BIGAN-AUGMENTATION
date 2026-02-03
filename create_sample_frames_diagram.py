"""
Create diagram showing sample frames from C3D-GAN generated videos
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def extract_frames_from_video(video_path, num_frames=5):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
    
    # Get evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    
    cap.release()
    return frames

def create_sample_frames_diagram():
    """Create a diagram showing sample frames from different videos"""
    video_dir = Path("final_videos")
    
    # Select sample videos with different conditions
    sample_videos = [
        "synth_0000_sexF_age0-1y_bmiunderweight_FINAL.mp4",
        "synth_0001_sexF_age0-1y_bminormal_FINAL.mp4",
        "synth_0003_sexF_age0-1y_bmiobese_FINAL.mp4",
        "synth_0005_sexF_age2-5y_bminormal_FINAL.mp4",
        "synth_0020_sexM_age0-1y_bmiunderweight.mp4",
    ]
    
    # Extract frames from each video
    all_frames = []
    labels = []
    
    for video_name in sample_videos:
        video_path = video_dir / video_name
        if video_path.exists():
            frames = extract_frames_from_video(video_path, num_frames=5)
            if frames:
                all_frames.append(frames)
                # Extract label from filename
                label = video_name.replace("synth_", "").replace("_FINAL.mp4", "").replace(".mp4", "")
                labels.append(label)
    
    if not all_frames:
        print("No videos found! Checking available videos...")
        videos = list(video_dir.glob("*.mp4"))
        print(f"Found {len(videos)} videos")
        if videos:
            # Use first few videos
            for video_path in videos[:5]:
                frames = extract_frames_from_video(video_path, num_frames=5)
                if frames:
                    all_frames.append(frames)
                    labels.append(video_path.stem)
    
    if not all_frames:
        print("Error: Could not extract frames from any videos")
        return
    
    # Create figure
    num_videos = len(all_frames)
    num_frames_per_video = len(all_frames[0])
    
    fig, axes = plt.subplots(num_videos, num_frames_per_video, 
                             figsize=(15, 3 * num_videos))
    
    if num_videos == 1:
        axes = axes.reshape(1, -1)
    if num_frames_per_video == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (frames, label) in enumerate(zip(all_frames, labels)):
        for j, frame in enumerate(frames):
            ax = axes[i, j] if num_videos > 1 else axes[j]
            ax.imshow(frame, cmap='gray')
            ax.axis('off')
            
            # Add label on first frame of each row
            if j == 0:
                ax.text(-10, frame.shape[0]//2, label, 
                       rotation=90, va='center', ha='right',
                       fontsize=10, fontweight='bold')
            
            # Add frame number on top
            if i == 0:
                ax.text(frame.shape[1]//2, -5, f'Frame {j+1}', 
                       ha='center', va='top', fontsize=9)
    
    plt.suptitle('C3D-GAN Generated Video Sample Frames', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('c3dgan_sample_frames_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved sample frames diagram to: c3dgan_sample_frames_diagram.png")
    plt.close()

if __name__ == "__main__":
    create_sample_frames_diagram()







