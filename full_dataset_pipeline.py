"""
FULL DATASET PIPELINE: Process entire EchoNet-Pediatric dataset
This script processes ALL videos (not just 2000) for high-quality C3DGAN training
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm == 0:
        return np.nan
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def categorize_age(age):
    """Categorize age into bins"""
    if pd.isna(age):
        return "unknown"
    age = float(age)
    if age < 1:
        return "0-1"
    elif age < 6:
        return "2-5"
    elif age < 11:
        return "6-10"
    elif age < 16:
        return "11-15"
    elif age <= 18:
        return "16-18"
    else:
        return "unknown"


def categorize_bmi_percentile(age, bmi, sex):
    """Categorize BMI into percentile bins for pediatric patients"""
    if pd.isna(bmi) or pd.isna(age):
        return "unknown"
    
    age = float(age)
    
    if age < 2:
        if bmi < 14:
            return "underweight"
        elif bmi < 18:
            return "normal"
        elif bmi < 20:
            return "overweight"
        else:
            return "obese"
    else:
        if bmi < 15:
            return "underweight"
        elif bmi < 20:
            return "normal"
        elif bmi < 25:
            return "overweight"
        else:
            return "obese"


def load_full_dataset(dataset_path=None):
    """Load ALL videos from A4C and PSAX datasets"""
    if dataset_path is None:
        dataset_path = "Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("\nPlease ensure the dataset is downloaded.")
        print("Expected structure:")
        print("  Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/")
        print("    ├── A4C/")
        print("    │   ├── FileList.csv")
        print("    │   └── Videos/")
        print("    └── PSAX/")
        print("        ├── FileList.csv")
        print("        └── Videos/")
        return None
    
    # Load A4C
    a4c_file_list = os.path.join(dataset_path, "A4C/FileList.csv")
    if not os.path.exists(a4c_file_list):
        print(f"❌ A4C FileList.csv not found: {a4c_file_list}")
        return None
    
    a4c_df = pd.read_csv(a4c_file_list)
    a4c_df['view'] = 'A4C'
    a4c_df['video_path'] = a4c_df['FileName'].apply(
        lambda x: os.path.join(dataset_path, f"A4C/Videos/{x}")
    )
    
    # Load PSAX
    psax_file_list = os.path.join(dataset_path, "PSAX/FileList.csv")
    if not os.path.exists(psax_file_list):
        print(f"❌ PSAX FileList.csv not found: {psax_file_list}")
        return None
    
    psax_df = pd.read_csv(psax_file_list)
    psax_df['view'] = 'PSAX'
    psax_df['video_path'] = psax_df['FileName'].apply(
        lambda x: os.path.join(dataset_path, f"PSAX/Videos/{x}")
    )
    
    # Combine
    df = pd.concat([a4c_df, psax_df], ignore_index=True)
    
    print(f"\n✅ Total videos in dataset: {len(df)}")
    print(f"   A4C: {len(a4c_df)}")
    print(f"   PSAX: {len(psax_df)}")
    
    return df


def filter_valid_videos(df):
    """Filter videos with complete metadata and existing files"""
    print("\nFiltering videos with complete data...")
    
    initial_count = len(df)
    
    # Filter missing data
    df_clean = df[
        df['Sex'].notna() & 
        df['Age'].notna() & 
        df['Weight'].notna() & 
        df['Height'].notna()
    ].copy()
    
    print(f"   Videos with complete metadata: {len(df_clean)} (removed {initial_count - len(df_clean)})")
    
    # Check file existence
    print("   Checking file existence...")
    existing_mask = df_clean['video_path'].apply(os.path.exists)
    df_clean = df_clean[existing_mask].reset_index(drop=True)
    
    print(f"   Videos with existing files: {len(df_clean)}")
    
    # Calculate BMI
    df_clean['BMI'] = df_clean.apply(
        lambda row: calculate_bmi(row['Weight'], row['Height']),
        axis=1
    )
    
    # Categorize
    df_clean['age_bin'] = df_clean['Age'].apply(categorize_age)
    df_clean['bmi_category'] = df_clean.apply(
        lambda row: categorize_bmi_percentile(row['Age'], row['BMI'], row['Sex']),
        axis=1
    )
    
    # Remove unknowns
    df_clean = df_clean[
        (df_clean['age_bin'] != 'unknown') & 
        (df_clean['bmi_category'] != 'unknown')
    ].reset_index(drop=True)
    
    print(f"   Videos with valid categories: {len(df_clean)}")
    
    # Show distribution
    print("\n   Distribution:")
    print(f"     View: {dict(df_clean['view'].value_counts())}")
    print(f"     Sex: {dict(df_clean['Sex'].value_counts())}")
    print(f"     Age: {dict(df_clean['age_bin'].value_counts())}")
    print(f"     BMI: {dict(df_clean['bmi_category'].value_counts())}")
    
    return df_clean


def preprocess_video(input_path, output_path, target_size=64, target_frames=32):
    """
    Preprocess video and save as numpy array
    Returns True if successful
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False
        
        # Sample frame indices
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        
        # Read frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (target_size, target_size))
            frames.append(resized)
        
        cap.release()
        
        if len(frames) != target_frames:
            return False
        
        # Save as numpy array
        arr = np.stack(frames, axis=0).astype(np.uint8)  # [T, H, W]
        np.save(output_path, arr)
        
        return True
        
    except Exception as e:
        return False


def preprocess_dataset(df, output_dir="data_numpy_full", size=64, frames=32, resume=False):
    """
    Preprocess all videos and save as numpy arrays
    Supports resuming from previous run
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING {len(df)} VIDEOS")
    print(f"{'='*70}")
    print(f"Target size: {size}x{size}, Frames: {frames}")
    print(f"Output directory: {output_dir}")
    print(f"Resume mode: {resume}")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing manifest if resuming
    manifest_path = os.path.join(output_dir, "manifest.csv")
    existing_processed = set()
    if resume and os.path.exists(manifest_path):
        try:
            existing_df = pd.read_csv(manifest_path)
            existing_processed = set(existing_df['processed_path'].apply(lambda x: os.path.basename(x)))
            print(f"✓ Resuming: Found {len(existing_processed)} already processed videos")
        except:
            pass
    
    successful = []
    failed = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        input_path = row['video_path']
        filename = Path(input_path).stem + '.npy'
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already processed
        if resume and filename in existing_processed:
            skipped += 1
            continue
        
        if preprocess_video(input_path, output_path, size, frames):
            row_copy = row.copy()
            row_copy['processed_path'] = output_path
            row_copy['processed_size'] = size
            row_copy['processed_frames'] = frames
            successful.append(row_copy)
        else:
            failed.append(input_path)
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {len(successful)}")
    if skipped > 0:
        print(f"⏭️  Skipped (already processed): {skipped}")
    print(f"❌ Failed: {len(failed)}")
    
    # Load existing manifest and merge
    if resume and os.path.exists(manifest_path):
        try:
            existing_df = pd.read_csv(manifest_path)
            successful_df = pd.concat([existing_df, pd.DataFrame(successful)], ignore_index=True)
            successful_df = successful_df.drop_duplicates(subset=['processed_path'], keep='last')
        except:
            successful_df = pd.DataFrame(successful)
    else:
        successful_df = pd.DataFrame(successful)
    
    # Save manifest
    successful_df.to_csv(manifest_path, index=False)
    print(f"\n✅ Manifest saved: {manifest_path}")
    print(f"   Total videos in manifest: {len(successful_df)}")
    
    if len(failed) > 0:
        failed_path = os.path.join(output_dir, "failed_videos.txt")
        with open(failed_path, 'w') as f:
            for path in failed:
                f.write(f"{path}\n")
        print(f"⚠️  Failed videos list: {failed_path}")
    
    return successful_df


def main():
    parser = argparse.ArgumentParser(description="Process FULL EchoNet-Pediatric dataset")
    parser.add_argument("--dataset_path", type=str, default=None, 
                       help="Path to dataset root (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default="data_numpy_full",
                       help="Output directory for processed videos")
    parser.add_argument("--size", type=int, default=64, choices=[32, 64, 128],
                       help="Target video size (default: 64)")
    parser.add_argument("--frames", type=int, default=32,
                       help="Number of frames per video (default: 32)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run (skip already processed videos)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL DATASET PREPROCESSING PIPELINE")
    print("="*70)
    
    # 1. Load full dataset
    print("\n[1/3] Loading FULL dataset...")
    df_raw = load_full_dataset(args.dataset_path)
    
    if df_raw is None:
        print("\n❌ Failed to load dataset. Please check the dataset path.")
        return
    
    # 2. Filter valid videos
    print("\n[2/3] Filtering valid videos...")
    df_valid = filter_valid_videos(df_raw)
    
    if len(df_valid) == 0:
        print("\n❌ No valid videos found!")
        return
    
    # 3. Preprocess all videos
    print("\n[3/3] Preprocessing videos...")
    df_processed = preprocess_dataset(
        df_valid, 
        output_dir=args.output_dir,
        size=args.size,
        frames=args.frames,
        resume=args.resume
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   Total videos in dataset: {len(df_raw)}")
    print(f"   Valid videos: {len(df_valid)}")
    print(f"   Successfully processed: {len(df_processed)}")
    print(f"\n📁 Output:")
    print(f"   Directory: {args.output_dir}/")
    print(f"   Manifest: {args.output_dir}/manifest.csv")
    print(f"   Format: numpy arrays (*.npy)")
    print(f"   Resolution: {args.size}x{args.size}x{args.frames}")
    
    print("\n✅ Ready for C3DGAN training!")
    print(f"Next step: Train C3DGAN with:")
    print(f"  python final_pipeline/train_c3dgan.py --manifest {args.output_dir}/manifest.csv --epochs 50 --batch_size 16 --size {args.size}")
    print("="*70)


if __name__ == "__main__":
    main()

