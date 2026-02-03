"""
Create enhanced visualizations for final report:
1. Per-condition coverage table
2. t-SNE/UMAP embedding plot (real vs synthetic)
3. Detailed distribution comparisons
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

def create_condition_coverage_table():
    """Create detailed table of samples per condition combination"""
    print("Creating per-condition coverage table...")
    
    files = glob.glob('final_videos/*.npy')
    
    # Parse all conditions
    data = []
    for f in files:
        fname = Path(f).stem
        
        # Extract conditions
        sex = 'F' if 'sexF' in fname else 'M'
        
        age = None
        for a in ['0-1y', '2-5y', '6-10y', '11-15y', '16-18y']:
            if a in fname:
                age = a
                break
        
        bmi = None
        for b in ['underweight', 'normal', 'overweight', 'obese']:
            if b in fname:
                bmi = b
                break
        
        if age and bmi:
            data.append({'Sex': sex, 'Age': age, 'BMI': bmi, 'File': fname})
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    coverage = df.groupby(['Sex', 'Age', 'BMI']).size().reset_index(name='Count')
    
    # Save to CSV
    coverage.to_csv('validation_report/condition_coverage.csv', index=False)
    print(f"  ✓ Saved: validation_report/condition_coverage.csv")
    
    # Print summary
    print(f"\n  Total combinations covered: {len(coverage)}/40")
    print(f"  Min samples per combo: {coverage['Count'].min()}")
    print(f"  Max samples per combo: {coverage['Count'].max()}")
    print(f"  Mean samples per combo: {coverage['Count'].mean():.1f}")
    
    return coverage


def create_embedding_visualization():
    """Create t-SNE visualization of real vs synthetic videos"""
    print("\nCreating t-SNE embedding visualization...")
    
    # Load videos
    print("  Loading real videos...")
    real_files = list(Path('data_numpy').glob('*.npy'))[:100]
    real_videos = []
    for f in tqdm(real_files[:50], desc="  Real", leave=False):
        video = np.load(f)
        # Use middle frame as feature
        real_videos.append(video[16].flatten())
    
    print("  Loading synthetic videos...")
    synth_files = list(Path('final_videos').glob('*.npy'))[:100]
    synth_videos = []
    for f in tqdm(synth_files[:50], desc="  Synthetic", leave=False):
        video = np.load(f)
        # Use middle frame as feature
        synth_videos.append(video[16].flatten())
    
    # Combine
    all_videos = np.array(real_videos + synth_videos)
    labels = ['Real'] * len(real_videos) + ['Synthetic'] * len(synth_videos)
    
    # Reduce dimensionality with PCA first (for speed)
    print("  Reducing dimensionality...")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(all_videos)
    
    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(reduced)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for label, color in [('Real', 'blue'), ('Synthetic', 'orange')]:
        mask = np.array(labels) == label
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=color, label=label, alpha=0.6, s=50)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Embedding: Real vs Synthetic Videos\n(Middle frame features)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_report/tsne_embedding.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: validation_report/tsne_embedding.png")
    
    # Calculate overlap metric
    real_center = embedded[:len(real_videos)].mean(axis=0)
    synth_center = embedded[len(real_videos):].mean(axis=0)
    center_distance = np.linalg.norm(real_center - synth_center)
    
    print(f"\n  Center distance: {center_distance:.2f}")
    print(f"  Interpretation: {'✅ Good overlap' if center_distance < 20 else '⚠️ Separated clusters'}")


def create_detailed_distributions():
    """Create detailed distribution comparison plots"""
    print("\nCreating detailed distribution plots...")
    
    # Load data
    real_videos = []
    for f in tqdm(list(Path('data_numpy').glob('*.npy'))[:100], desc="  Loading real", leave=False):
        real_videos.append(np.load(f))
    real_videos = np.array(real_videos)
    
    synth_videos = []
    for f in tqdm(list(Path('final_videos').glob('*.npy'))[:100], desc="  Loading synthetic", leave=False):
        synth_videos.append(np.load(f))
    synth_videos = np.array(synth_videos)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Statistical Comparison: Real vs Synthetic', fontsize=16, fontweight='bold')
    
    # 1. Intensity distribution (improved)
    ax = axes[0, 0]
    ax.hist(real_videos.flatten(), bins=50, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(synth_videos.flatten(), bins=50, alpha=0.6, label='Synthetic', color='orange', density=True)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Intensity Distribution (Normalized)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Per-frame mean intensity
    ax = axes[0, 1]
    real_means = real_videos.mean(axis=(0, 2, 3))
    synth_means = synth_videos.mean(axis=(0, 2, 3))
    ax.plot(real_means, label='Real', color='blue', linewidth=2)
    ax.plot(synth_means, label='Synthetic', color='orange', linewidth=2)
    ax.fill_between(range(len(real_means)), real_means, alpha=0.3, color='blue')
    ax.fill_between(range(len(synth_means)), synth_means, alpha=0.3, color='orange')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('Temporal Intensity Profile')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Motion magnitude distribution
    ax = axes[1, 0]
    real_motion = np.abs(np.diff(real_videos, axis=1)).mean(axis=(1, 2, 3))
    synth_motion = np.abs(np.diff(synth_videos, axis=1)).mean(axis=(1, 2, 3))
    ax.hist(real_motion, bins=30, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(synth_motion, bins=30, alpha=0.6, label='Synthetic', color='orange', density=True)
    ax.set_xlabel('Motion Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Motion Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Variance comparison
    ax = axes[1, 1]
    real_var = real_videos.var(axis=1).flatten()
    synth_var = synth_videos.var(axis=1).flatten()
    ax.hist(real_var, bins=30, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(synth_var, bins=30, alpha=0.6, label='Synthetic', color='orange', density=True)
    ax.set_xlabel('Temporal Variance')
    ax.set_ylabel('Density')
    ax.set_title('Temporal Variance Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_report/detailed_distributions.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: validation_report/detailed_distributions.png")


def main():
    print("="*70)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*70)
    
    Path('validation_report').mkdir(exist_ok=True)
    
    # 1. Coverage table
    coverage = create_condition_coverage_table()
    
    # 2. t-SNE embedding
    create_embedding_visualization()
    
    # 3. Detailed distributions
    create_detailed_distributions()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("="*70)
    print("\nGenerated files:")
    print("  • validation_report/condition_coverage.csv")
    print("  • validation_report/tsne_embedding.png")
    print("  • validation_report/detailed_distributions.png")
    print("\nOpen folder:")
    print("  open validation_report/")
    print("="*70)


if __name__ == "__main__":
    main()






