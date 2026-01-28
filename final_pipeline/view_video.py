"""
View generated videos saved as numpy arrays
"""
import argparse
import numpy as np
import cv2
import os
import glob


def view_video(npy_path, fps=30, loop=True):
    """Display a video from numpy array"""
    # Load video
    video = np.load(npy_path)  # [T, H, W]
    
    print(f"Video: {os.path.basename(npy_path)}")
    print(f"Shape: {video.shape}")
    print(f"Range: [{video.min()}, {video.max()}]")
    
    window_name = os.path.basename(npy_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_delay = int(1000 / fps)
    
    while True:
        for t in range(video.shape[0]):
            frame = video[t]
            
            # Resize for better viewing
            display_frame = cv2.resize(frame, (256, 256))
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(frame_delay)
            if key == ord('q') or key == 27:  # q or ESC
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):  # next video
                cv2.destroyAllWindows()
                return
        
        if not loop:
            break
    
    cv2.destroyAllWindows()


def view_all_videos(directory, fps=30):
    """View all videos in a directory"""
    npy_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return
    
    print(f"Found {len(npy_files)} videos")
    print("Controls: Q/ESC=quit, N=next video\n")
    
    for npy_file in npy_files:
        view_video(npy_file, fps=fps, loop=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View generated videos")
    parser.add_argument("path", type=str, help="Path to .npy file or directory")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop single video")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        view_all_videos(args.path, fps=args.fps)
    else:
        view_video(args.path, fps=args.fps, loop=args.loop)






