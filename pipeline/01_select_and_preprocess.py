"""
Step 1: Select 2000 high-quality videos and preprocess with BMI calculation
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import shutil

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
    """
    Categorize BMI into percentile bins for pediatric patients
    Simplified bins: underweight, normal, overweight, obese
    """
    if pd.isna(bmi) or pd.isna(age):
        return "unknown"
    
    # Simplified pediatric BMI categories (approximations)
    # Actual percentiles would need CDC growth charts
    age = float(age)
    
    if age < 2:
        # For infants, different criteria
        if bmi < 14:
            return "underweight"
        elif bmi < 18:
            return "normal"
        elif bmi < 20:
            return "overweight"
        else:
            return "obese"
    else:
        # For children 2-18
        if bmi < 15:
            return "underweight"
        elif bmi < 20:
            return "normal"
        elif bmi < 25:
            return "overweight"
        else:
            return "obese"

def load_raw_dataset():
    """Load raw A4C and PSAX datasets"""
    dataset_path = "Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi"
    
    # Load A4C
    a4c_file_list = os.path.join(dataset_path, "A4C/FileList.csv")
    a4c_df = pd.read_csv(a4c_file_list)
    a4c_df['view'] = 'A4C'
    a4c_df['video_path'] = a4c_df['FileName'].apply(
        lambda x: os.path.join(dataset_path, f"A4C/Videos/{x}")
    )
    
    # Load PSAX
    psax_file_list = os.path.join(dataset_path, "PSAX/FileList.csv")
    psax_df = pd.read_csv(psax_file_list)
    psax_df['view'] = 'PSAX'
    psax_df['video_path'] = psax_df['FileName'].apply(
        lambda x: os.path.join(dataset_path, f"PSAX/Videos/{x}")
    )
    
    # Combine
    df = pd.concat([a4c_df, psax_df], ignore_index=True)
    
    print(f"Total videos in dataset: {len(df)}")
    print(f"  A4C: {len(a4c_df)}")
    print(f"  PSAX: {len(psax_df)}")
    
    return df

def select_quality_videos(df, n=2000):
    """
    Select 2000 high-quality videos with good distribution
    """
    print(f"\nSelecting {n} high-quality videos...")
    
    # Filter out missing data
    df_clean = df[
        df['Sex'].notna() & 
        df['Age'].notna() & 
        df['Weight'].notna() & 
        df['Height'].notna() &
        df['video_path'].apply(os.path.exists)
    ].copy()
    
    print(f"Videos with complete data: {len(df_clean)}")
    
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
    ]
    
    print(f"Videos with valid categories: {len(df_clean)}")
    
    # Stratified sampling for balanced distribution
    # Try to get equal representation across categories
    samples = []
    
    # Group by view, sex, age_bin, bmi_category
    grouped = df_clean.groupby(['view', 'Sex', 'age_bin', 'bmi_category'])
    
    # Calculate samples per group
    n_groups = len(grouped)
    samples_per_group = max(1, n // n_groups)
    
    print(f"Number of groups: {n_groups}")
    print(f"Target samples per group: {samples_per_group}")
    
    for name, group in grouped:
        # Sample from each group
        n_sample = min(samples_per_group, len(group))
        sampled = group.sample(n=n_sample, random_state=42)
        samples.append(sampled)
    
    selected_df = pd.concat(samples, ignore_index=True)
    
    # If we have more than needed, sample down
    if len(selected_df) > n:
        selected_df = selected_df.sample(n=n, random_state=42)
    
    print(f"\nSelected {len(selected_df)} videos")
    
    # Distribution summary
    print("\nDistribution by view:")
    print(selected_df['view'].value_counts())
    print("\nDistribution by sex:")
    print(selected_df['Sex'].value_counts())
    print("\nDistribution by age bin:")
    print(selected_df['age_bin'].value_counts())
    print("\nDistribution by BMI category:")
    print(selected_df['bmi_category'].value_counts())
    
    return selected_df

def preprocess_video(input_path, output_path, target_size=64, target_frames=32):
    """
    Preprocess a single video:
    - Resize to target_size x target_size
    - Sample target_frames uniformly
    - Convert to grayscale
    - Save as MP4
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False
        
        # Sample frame indices uniformly
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        
        # Read and process frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize
            resized = cv2.resize(gray, (target_size, target_size))
            
            frames.append(resized)
        
        cap.release()
        
        if len(frames) != target_frames:
            return False
        
        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (target_size, target_size), isColor=False)
        
        for frame in frames:
            # Convert back to BGR for writing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def preprocess_dataset(df, output_dir="data_fresh/videos", size=64, frames=32):
    """Preprocess all selected videos"""
    print(f"\nPreprocessing {len(df)} videos...")
    print(f"Target size: {size}x{size}, Frames: {frames}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful = []
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        input_path = row['video_path']
        filename = Path(input_path).stem + '.mp4'
        output_path = os.path.join(output_dir, filename)
        
        if preprocess_video(input_path, output_path, size, frames):
            row_copy = row.copy()
            row_copy['processed_path'] = output_path
            row_copy['processed_size'] = size
            row_copy['processed_frames'] = frames
            successful.append(row_copy)
        else:
            failed.append(input_path)
    
    print(f"\nSuccessfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(failed) > 0:
        print(f"Failed videos (first 10): {failed[:10]}")
    
    return pd.DataFrame(successful)

def main():
    print("="*70)
    print("STEP 1: SELECT AND PREPROCESS DATASET")
    print("="*70)
    
    # 1. Load raw dataset
    print("\n[1/4] Loading raw dataset...")
    df_raw = load_raw_dataset()
    
    # 2. Select 2000 quality videos
    print("\n[2/4] Selecting 2000 high-quality videos...")
    df_selected = select_quality_videos(df_raw, n=2000)
    
    # 3. Preprocess videos
    print("\n[3/4] Preprocessing videos...")
    df_processed = preprocess_dataset(df_selected, output_dir="data_fresh/videos", size=64, frames=32)
    
    # 4. Save manifest
    print("\n[4/4] Saving manifest...")
    os.makedirs("data_fresh", exist_ok=True)
    manifest_path = "data_fresh/manifest.csv"
    df_processed.to_csv(manifest_path, index=False)
    print(f"Saved manifest to: {manifest_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nTotal videos processed: {len(df_processed)}")
    print(f"Output directory: data_fresh/videos/")
    print(f"Manifest: {manifest_path}")
    
    print("\nFinal distribution:")
    print("\nBy Sex:")
    print(df_processed['Sex'].value_counts())
    print("\nBy Age Bin:")
    print(df_processed['age_bin'].value_counts())
    print("\nBy BMI Category:")
    print(df_processed['bmi_category'].value_counts())
    print("\nBy View:")
    print(df_processed['view'].value_counts())
    
    print("\n" + "="*70)
    print("✅ Ready for BiGAN training!")
    print("Next step: python pipeline/02_train_bigan.py")
    print("="*70)

if __name__ == "__main__":
    main()

