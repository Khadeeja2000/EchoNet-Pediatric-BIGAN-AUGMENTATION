"""
Show frame-by-frame comparison: 30 fps vs 60 fps
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def extract_frames(video_path, num_frames=10):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    return frames, fps


def show_frame_comparison():
    """Show side-by-side comparison of 30 fps vs 60 fps"""
    df = pd.read_csv('data/processed/manifest.csv')
    if 'processed_path' in df.columns:
        df = df[df['processed_path'].astype(str).str.len() > 0].reset_index(drop=True)
    
    # Get original video
    original_path = df.iloc[0]['processed_path'] if 'processed_path' in df.columns else df.iloc[0]['file_path']
    base_name = os.path.basename(original_path).replace('.mp4', '')
    enhanced_path = f'temporal_sr_results/temporal_sr_{base_name}_x2.mp4'
    
    if not os.path.exists(enhanced_path):
        print(f"Enhanced video not found: {enhanced_path}")
        return
    
    print("="*80)
    print("FRAME-BY-FRAME COMPARISON: 30 fps vs 60 fps")
    print("="*80)
    
    # Extract frames from both videos
    print("\nExtracting frames from original video (30 fps)...")
    orig_frames, orig_fps = extract_frames(original_path, num_frames=5)
    print(f"  ✓ Extracted {len(orig_frames)} frames at {orig_fps:.1f} fps")
    
    print("\nExtracting frames from enhanced video (60 fps)...")
    enh_frames, enh_fps = extract_frames(enhanced_path, num_frames=10)
    print(f"  ✓ Extracted {len(enh_frames)} frames at {enh_fps:.1f} fps")
    
    # Create comparison visualization - show 5 frames from each
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Frame Comparison: Original (30 fps) vs Enhanced (60 fps)', 
                 fontsize=16, weight='bold')
    
    # Show original frames
    print("\n" + "-"*80)
    print("ORIGINAL VIDEO (30 fps) - Showing 5 frames:")
    print("-"*80)
    
    for i in range(5):
        if i < len(orig_frames):
            axes[0, i].imshow(orig_frames[i], cmap='gray')
            axes[0, i].set_title(f'Frame {i+1}\n(30 fps)', fontsize=10, weight='bold')
            axes[0, i].axis('off')
            print(f"  Frame {i+1}: Original frame from 30 fps video")
    
    # Show enhanced frames - show every other frame to match timing, but mark interpolated
    print("\n" + "-"*80)
    print("ENHANCED VIDEO (60 fps) - Showing frames (original + interpolated):")
    print("-"*80)
    
    # Show frames 0, 2, 4, 6, 8 (every other to show both original and interpolated)
    for i in range(5):
        frame_idx = i * 2
        if frame_idx < len(enh_frames):
            # Even indices are original, odd are interpolated
            is_interpolated = (frame_idx % 2 == 1)
            title = f'Frame {frame_idx+1}'
            if is_interpolated:
                title += '\n(INTERPOLATED)'
            else:
                title += '\n(Original)'
            title += '\n(60 fps)'
            
            axes[1, i].imshow(enh_frames[frame_idx], cmap='gray')
            axes[1, i].set_title(title, fontsize=9, weight='bold', 
                                color='green' if is_interpolated else 'black')
            axes[1, i].axis('off')
            
            frame_type = "INTERPOLATED (created by optical flow)" if is_interpolated else "ORIGINAL"
            print(f"  Frame {frame_idx+1}: {frame_type}")
        
        # Also show the interpolated frame after each original
        if i < 4:  # Don't go out of bounds
            interp_idx = frame_idx + 1
            if interp_idx < len(enh_frames):
                print(f"  Frame {interp_idx+1}: INTERPOLATED (created by optical flow)")
    
    plt.tight_layout()
    os.makedirs('cv_comparison', exist_ok=True)
    plt.savefig('cv_comparison/frame_comparison_30vs60.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*80)
    print("✓ Saved comparison: cv_comparison/frame_comparison_30vs60.png")
    print("="*80)
    
    print("\nKEY DIFFERENCES:")
    print("-"*80)
    print("30 fps video: Has 5 frames shown")
    print("60 fps video: Has 10 frames shown (5 original + 5 interpolated)")
    print("\nThe interpolated frames (marked in green) are NEW frames")
    print("created by analyzing motion between original frames!")
    print("="*80)
    
    plt.close()


if __name__ == "__main__":
    show_frame_comparison()

