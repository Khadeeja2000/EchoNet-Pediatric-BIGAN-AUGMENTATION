"""
COMPUTER VISION PROJECT: Temporal Super-Resolution (Frame Interpolation)
Uses optical flow to interpolate frames and increase video frame rate
Classic computer vision technique - perfect for CV course project
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


def compute_optical_flow(frame1, frame2):
    """Compute dense optical flow between two frames"""
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        gray2 = frame2
    
    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5,      # Image pyramid scale
        levels=3,            # Number of pyramid levels
        winsize=15,         # Averaging window size
        iterations=3,        # Iterations at each pyramid level
        poly_n=5,            # Polynomial expansion neighborhood
        poly_sigma=1.2,     # Gaussian sigma for polynomial expansion
        flags=0             # Flow flags
    )
    
    return flow


def interpolate_frame(frame1, frame2, flow, alpha=0.5):
    """Interpolate a frame between two frames using optical flow"""
    h, w = frame1.shape[:2]
    
    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Forward flow: where pixels in frame1 move to in frame2
    flow_forward = flow
    
    # Backward flow: where pixels in frame2 come from in frame1
    flow_backward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2,
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1,
        None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Forward warping: move frame1 forward
    x1 = x + alpha * flow_forward[:, :, 0]
    y1 = y + alpha * flow_forward[:, :, 1]
    
    # Backward warping: move frame2 backward
    x2 = x + (1 - alpha) * flow_backward[:, :, 0]
    y2 = y + (1 - alpha) * flow_backward[:, :, 1]
    
    # Create masks for valid pixels
    mask1 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
    mask2 = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
    
    # Initialize interpolated frame
    if len(frame1.shape) == 3:
        interpolated = np.zeros_like(frame1, dtype=np.float32)
    else:
        interpolated = np.zeros_like(frame1, dtype=np.float32)
    
    # Forward warping
    x1_valid = np.clip(x1, 0, w - 1).astype(np.int32)
    y1_valid = np.clip(y1, 0, h - 1).astype(np.int32)
    interpolated[mask1] = (1 - alpha) * frame1[y1_valid[mask1], x1_valid[mask1]]
    
    # Backward warping
    x2_valid = np.clip(x2, 0, w - 1).astype(np.int32)
    y2_valid = np.clip(y2, 0, h - 1).astype(np.int32)
    interpolated[mask2] += alpha * frame2[y2_valid[mask2], x2_valid[mask2]]
    
    # Average where both are valid
    both_valid = mask1 & mask2
    interpolated[both_valid] = 0.5 * (
        (1 - alpha) * frame1[y1_valid[both_valid], x1_valid[both_valid]] +
        alpha * frame2[y2_valid[both_valid], x2_valid[both_valid]]
    )
    
    return np.clip(interpolated, 0, 255).astype(np.uint8)


def temporal_super_resolution(video_path, output_path, factor=2):
    """
    Increase video frame rate using temporal super-resolution
    
    Args:
        video_path: Input video path
        output_path: Output video path
        factor: Frame rate multiplier (2 = double frame rate)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Increasing frame rate by factor {factor}...")
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps * factor
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        cap.release()
        return
    
    # Convert to grayscale for processing (will convert back for output)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    
    frame_count = 0
    interpolated_count = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        
        # Write original frame
        out.write(prev_frame)
        frame_count += 1
        
        # Interpolate frames between prev and curr
        for i in range(1, factor):
            alpha = i / factor
            interpolated = interpolate_frame(prev_frame, curr_frame, 
                                           compute_optical_flow(prev_gray, curr_gray), 
                                           alpha=alpha)
            out.write(interpolated)
            interpolated_count += 1
        
        prev_frame = curr_frame
        prev_gray = curr_gray
    
    # Write last frame
    out.write(prev_frame)
    frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"✓ Complete!")
    print(f"  Original frames: {frame_count}")
    print(f"  Interpolated frames: {interpolated_count}")
    print(f"  Total output frames: {frame_count + interpolated_count}")
    print(f"  Output FPS: {out_fps:.1f}")
    print(f"  Saved to: {output_path}")


def visualize_optical_flow(frame1, frame2, flow, output_path):
    """Visualize optical flow as a color-coded image"""
    h, w = flow.shape[:2]
    
    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    
    # Convert to BGR for display
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Save visualization
    cv2.imwrite(output_path, flow_vis)
    print(f"  Flow visualization saved: {output_path}")


def process_videos(manifest_csv, output_dir, num_samples=5, factor=2):
    """Process multiple videos for temporal super-resolution"""
    df = pd.read_csv(manifest_csv)
    if "processed_path" in df.columns:
        df = df[df["processed_path"].astype(str).str.len() > 0].reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "flow_visualizations"), exist_ok=True)
    
    print("="*60)
    print("TEMPORAL SUPER-RESOLUTION (Frame Interpolation)")
    print("="*60)
    print(f"Processing {num_samples} videos...")
    print(f"Frame rate multiplier: {factor}x")
    print("="*60)
    print()
    
    results = []
    
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        video_path = row["processed_path"] if "processed_path" in row else row["file_path"]
        
        if not os.path.exists(video_path):
            print(f"⚠ Video {i+1}: {video_path} not found, skipping...")
            continue
        
        print(f"Processing video {i+1}/{num_samples}...")
        
        # Output paths
        base_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"temporal_sr_{base_name}_x{factor}.mp4")
        flow_vis_path = os.path.join(output_dir, "flow_visualizations", f"flow_{base_name}.jpg")
        
        # Get original video info
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process video
        temporal_super_resolution(video_path, output_path, factor=factor)
        
        # Create flow visualization (first two frames)
        cap = cv2.VideoCapture(video_path)
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        if ret1 and ret2:
            flow = compute_optical_flow(frame1, frame2)
            visualize_optical_flow(frame1, frame2, flow, flow_vis_path)
        cap.release()
        
        results.append({
            "video_id": i,
            "input_path": video_path,
            "output_path": output_path,
            "flow_visualization": flow_vis_path,
            "original_fps": orig_fps,
            "output_fps": orig_fps * factor,
            "original_frames": orig_frames,
            "factor": factor
        })
        
        print()
    
    print("="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Processed {len(results)} videos")
    print(f"Results saved to: {output_dir}/")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Super-Resolution using Optical Flow")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--output_dir", type=str, default="temporal_sr_results")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--factor", type=int, default=2, help="Frame rate multiplier (2 = double)")
    args = parser.parse_args()
    
    process_videos(args.manifest, args.output_dir, args.num_samples, args.factor)




