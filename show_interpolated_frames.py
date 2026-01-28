"""
Show the actual interpolated frames clearly
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def show_interpolated_frames_clearly():
    """Show original vs interpolated frames side by side"""
    df = pd.read_csv('data/processed/manifest.csv')
    if 'processed_path' in df.columns:
        df = df[df['processed_path'].astype(str).str.len() > 0].reset_index(drop=True)
    
    original_path = df.iloc[0]['processed_path'] if 'processed_path' in df.columns else df.iloc[0]['file_path']
    base_name = os.path.basename(original_path).replace('.mp4', '')
    enhanced_path = f'temporal_sr_results/temporal_sr_{base_name}_x2.mp4'
    
    if not os.path.exists(enhanced_path):
        print("Enhanced video not found!")
        return
    
    print("="*80)
    print("SHOWING INTERPOLATED FRAMES")
    print("="*80)
    
    # Extract frames
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(enhanced_path)
    
    # Get first 3 frame pairs
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Frame Interpolation: Original → Interpolated → Next Original', 
                 fontsize=16, weight='bold')
    
    for row in range(3):
        # Original frame 1
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        axes[row, 0].imshow(gray1, cmap='gray')
        axes[row, 0].set_title(f'Original Frame {row+1}\n(30 fps)', 
                               fontsize=11, weight='bold', color='blue')
        axes[row, 0].axis('off')
        
        # Interpolated frame (from enhanced video)
        ret2, frame2 = cap2.read()  # Original
        ret2, interp_frame = cap2.read()  # Interpolated!
        if ret2:
            gray_interp = cv2.cvtColor(interp_frame, cv2.COLOR_BGR2GRAY)
            axes[row, 1].imshow(gray_interp, cmap='gray')
            axes[row, 1].set_title(f'INTERPOLATED Frame\n(60 fps - NEW!)', 
                                  fontsize=11, weight='bold', color='green')
            axes[row, 1].axis('off')
        
        # Original frame 2
        ret1, frame2_orig = cap1.read()
        if ret1:
            gray2 = cv2.cvtColor(frame2_orig, cv2.COLOR_BGR2GRAY)
            axes[row, 2].imshow(gray2, cmap='gray')
            axes[row, 2].set_title(f'Original Frame {row+2}\n(30 fps)', 
                                   fontsize=11, weight='bold', color='blue')
            axes[row, 2].axis('off')
        
        print(f"\nFrame Pair {row+1}:")
        print(f"  Original Frame {row+1} → INTERPOLATED Frame → Original Frame {row+2}")
        print(f"  (The middle frame is NEW - created by optical flow!)")
    
    cap1.release()
    cap2.release()
    
    plt.tight_layout()
    os.makedirs('cv_comparison', exist_ok=True)
    plt.savefig('cv_comparison/interpolated_frames_clear.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*80)
    print("✓ Saved: cv_comparison/interpolated_frames_clear.png")
    print("="*80)
    print("\nThis shows:")
    print("  - Left column: Original frames (30 fps)")
    print("  - Middle column: INTERPOLATED frames (created by optical flow)")
    print("  - Right column: Next original frames (30 fps)")
    print("\nThe middle frames are NEW - they didn't exist before!")
    print("They were created by analyzing motion between original frames.")
    print("="*80)
    
    plt.close()


if __name__ == "__main__":
    show_interpolated_frames_clearly()




