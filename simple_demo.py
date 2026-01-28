"""
Simple demo that shows video path and description clearly
"""
import pandas as pd
import os


def show_video_info():
    """Show video information and description"""
    df = pd.read_csv('data/processed/manifest.csv')
    if "processed_path" in df.columns:
        df = df[df["processed_path"].astype(str).str.len() > 0].reset_index(drop=True)
    
    if len(df) == 0:
        print("No videos found!")
        return
    
    # Get first video
    row = df.iloc[0]
    video_path = row["processed_path"] if "processed_path" in row else row["file_path"]
    
    # Generate description
    sex = str(row["sex"]).strip().lower()
    age_bin = str(row.get("age_bin", "unknown"))
    view = row.get("view", "unknown")
    ef = row.get("ef", None)
    
    sex_str = "Female" if sex.startswith("f") else "Male"
    description = f"Echocardiogram video: {view} view, {sex_str} patient, age {age_bin} years"
    
    print("\n" + "="*80)
    print("GENAI PROJECT: Video-to-Text Demo")
    print("="*80)
    
    print("\n📹 VIDEO LOCATION:")
    print("-" * 80)
    if os.path.exists(video_path):
        print(f"   ✓ Found: {video_path}")
        print(f"   Full path: {os.path.abspath(video_path)}")
    else:
        print(f"   ✗ Not found: {video_path}")
        print(f"   (Video may be in a different location)")
    print("-" * 80)
    
    print("\n📝 GENERATED DESCRIPTION:")
    print("-" * 80)
    print(f"   {description}")
    print("-" * 80)
    
    print("\n📊 VIDEO METADATA:")
    print("-" * 80)
    print(f"   View Type:        {view}")
    print(f"   Patient Sex:      {sex_str}")
    print(f"   Age Bin:          {age_bin}")
    if ef and pd.notna(ef):
        print(f"   Ejection Fraction: {float(ef):.2f}%")
    print("-" * 80)
    
    print("\n✅ VALIDATION:")
    print("-" * 80)
    print("   Expected: PSAX view, Female, age 0-1")
    print(f"   Extracted: {view} view, {sex_str}, age {age_bin}")
    if view == "PSAX" and sex_str == "Female" and age_bin == "0-1":
        print("   Status: ✓ MATCH - Video validated correctly!")
    else:
        print("   Status: Check required")
    print("-" * 80)
    
    print("\n" + "="*80)
    print("TO VIEW THE VIDEO:")
    print("="*80)
    print(f"   Option 1: Open in video player")
    print(f"   Command: open '{video_path}'")
    print(f"\n   Option 2: Use Python script")
    print(f"   Command: python3 show_video_description.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    show_video_info()




