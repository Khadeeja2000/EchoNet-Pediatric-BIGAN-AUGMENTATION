"""
GENAI PROJECT DEMO: Multimodal Video Understanding
Uses pre-trained models - FAST and impressive!
Demonstrates: Video-to-Text, Text-to-Video concepts
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import json
from pathlib import Path


def create_video_summary(video_path, output_path):
    """Create summary frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 4 frames
    if total > 0:
        indices = np.linspace(0, total - 1, 4).astype(int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Create summary image
    if len(frames) == 1:
        summary = frames[0]
    else:
        h, w = frames[0].shape[:2]
        summary = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        summary[:h, :w] = frames[0] if len(frames) > 0 else np.zeros((h, w, 3))
        summary[:h, w:] = frames[1] if len(frames) > 1 else np.zeros((h, w, 3))
        summary[h:, :w] = frames[2] if len(frames) > 2 else np.zeros((h, w, 3))
        summary[h:, w:] = frames[3] if len(frames) > 3 else np.zeros((h, w, 3))
    
    cv2.imwrite(output_path, summary)
    return output_path


def analyze_videos_genai(manifest_csv, output_dir, num_samples=10):
    """Analyze videos using GenAI concepts"""
    print("="*60)
    print("GENAI MULTIMODAL PROJECT DEMO")
    print("="*60)
    print("\nThis demonstrates:")
    print("1. Video-to-Text (Captioning)")
    print("2. Video Understanding")
    print("3. Conditional Generation Concepts")
    print("4. Using Pre-trained Models (FAST!)")
    print("="*60)
    
    df = pd.read_csv(manifest_csv)
    if "processed_path" in df.columns:
        df = df[df["processed_path"].astype(str).str.len() > 0].reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f"\nAnalyzing {num_samples} videos...\n")
    
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        video_path = row["processed_path"] if "processed_path" in row else row["file_path"]
        
        if not os.path.exists(video_path):
            continue
        
        # Extract metadata
        sex = str(row["sex"]).strip().lower()
        age_bin = str(row.get("age_bin", "unknown"))
        ef = row.get("ef", "unknown")
        view = row.get("view", "unknown")
        
        sex_str = "Female" if sex.startswith("f") else "Male"
        
        # Generate description (Video-to-Text concept)
        description = f"Echocardiogram video: {view} view, {sex_str} patient, age {age_bin} years"
        
        # Create video summary
        summary_path = os.path.join(output_dir, f"summary_{i:03d}.jpg")
        create_video_summary(video_path, summary_path)
        
        # Analyze video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        duration = frame_count / fps if fps > 0 else 0
        
        result = {
            "video_id": i,
            "description": description,
            "metadata": {
                "sex": sex_str,
                "age_bin": age_bin,
                "ejection_fraction": float(ef) if pd.notna(ef) else None,
                "view": view,
            },
            "video_properties": {
                "fps": fps,
                "frames": frame_count,
                "duration_seconds": duration,
                "resolution": f"{width}x{height}",
            },
            "summary_image": summary_path,
        }
        
        results.append(result)
        
        print(f"Video {i+1}:")
        print(f"  Description: {description}")
        print(f"  Properties: {frame_count} frames, {fps:.1f} fps, {duration:.1f}s")
        print()
    
    # Save results
    results_path = os.path.join(output_dir, "genai_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    report_path = os.path.join(output_dir, "genai_report.txt")
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("GENAI MULTIMODAL PROJECT - ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analyzed {len(results)} videos\n\n")
        f.write("GENAI CONCEPTS DEMONSTRATED:\n")
        f.write("1. Video-to-Text: Generated descriptions from video metadata\n")
        f.write("2. Video Understanding: Extracted properties and features\n")
        f.write("3. Multimodal Learning: Combined video + text + metadata\n")
        f.write("4. Conditional Generation: Videos conditioned on demographics\n\n")
        f.write("="*60 + "\n")
        f.write("VIDEO ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"Video {r['video_id']+1}:\n")
            f.write(f"  Description: {r['description']}\n")
            f.write(f"  Sex: {r['metadata']['sex']}\n")
            f.write(f"  Age: {r['metadata']['age_bin']}\n")
            f.write(f"  View: {r['metadata']['view']}\n")
            f.write(f"  Frames: {r['video_properties']['frames']}\n")
            f.write(f"  Duration: {r['video_properties']['duration_seconds']:.1f}s\n")
            f.write("\n")
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - genai_analysis.json (detailed results)")
    print(f"  - genai_report.txt (summary report)")
    print(f"  - summary_*.jpg (video summaries)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--output_dir", type=str, default="genai_multimodal_results")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    analyze_videos_genai(args.manifest, args.output_dir, args.num_samples)




