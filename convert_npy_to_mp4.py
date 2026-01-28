"""
Convert numpy videos to proper MP4 files that open in QuickTime
"""
import numpy as np
import imageio
import glob
import os
from tqdm import tqdm

def convert_npy_to_mp4(npy_path, mp4_path, fps=30):
    """Convert .npy video to proper MP4"""
    # Load numpy array
    video = np.load(npy_path)  # [T, H, W]
    
    # Convert to list of frames for imageio
    frames = []
    for t in range(video.shape[0]):
        # Convert grayscale to RGB for better compatibility
        frame = np.stack([video[t]] * 3, axis=-1)  # [H, W, 3]
        frames.append(frame)
    
    # Save with imageio (much better codec support)
    imageio.mimsave(mp4_path, frames, fps=fps, quality=8, codec='libx264')

# Convert all .npy files in test_generated
npy_files = glob.glob('test_generated/*.npy')

print(f"Converting {len(npy_files)} videos to MP4...")

for npy_file in tqdm(npy_files):
    mp4_file = npy_file.replace('.npy', '_playable.mp4')
    convert_npy_to_mp4(npy_file, mp4_file)

print(f"\n✅ Done! MP4 files saved with '_playable.mp4' suffix")
print(f"You can now double-click them to play in QuickTime!")






