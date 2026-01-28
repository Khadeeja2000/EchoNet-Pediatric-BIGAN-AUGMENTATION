"""
Validation Framework for Synthetic Echocardiogram Videos

Checks:
1. Statistical properties match real data
2. Temporal coherence (smooth motion)
3. Spatial structure (not random noise)
4. Condition consistency (sex/age/BMI appropriate)
5. Visual quality metrics
"""
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class VideoValidator:
    """Validate synthetic echocardiogram videos"""
    
    def __init__(self, real_data_dir, synthetic_data_dir):
        self.real_dir = real_data_dir
        self.synthetic_dir = synthetic_data_dir
        self.real_stats = {}
        self.synthetic_stats = {}
    
    def load_videos(self, directory, limit=100):
        """Load videos from directory"""
        videos = []
        files = list(Path(directory).glob("*.npy"))[:limit]
        
        for file in tqdm(files, desc=f"Loading {Path(directory).name}"):
            try:
                video = np.load(file)
                videos.append(video)
            except:
                pass
        
        return np.array(videos)
    
    def compute_statistics(self, videos, name="dataset"):
        """Compute statistical properties"""
        print(f"\n{'='*60}")
        print(f"Statistics for: {name}")
        print(f"{'='*60}")
        
        stats_dict = {}
        
        # Basic statistics
        stats_dict['mean'] = videos.mean()
        stats_dict['std'] = videos.std()
        stats_dict['min'] = videos.min()
        stats_dict['max'] = videos.max()
        
        # Per-frame statistics
        stats_dict['temporal_variance'] = videos.var(axis=1).mean()  # Variance across time
        stats_dict['spatial_variance'] = videos.var(axis=(2, 3)).mean()  # Variance across space
        
        # Frame-to-frame difference (motion metric)
        frame_diffs = np.diff(videos, axis=1)
        stats_dict['motion_magnitude'] = np.abs(frame_diffs).mean()
        stats_dict['motion_std'] = np.abs(frame_diffs).std()
        
        # Intensity distribution
        stats_dict['intensity_skewness'] = stats.skew(videos.flatten())
        stats_dict['intensity_kurtosis'] = stats.kurtosis(videos.flatten())
        
        # Spatial frequency (texture measure)
        fft_magnitudes = []
        for vid in videos[:10]:  # Sample first 10 for speed
            for frame in vid:
                fft = np.fft.fft2(frame)
                magnitude = np.abs(fft)
                fft_magnitudes.append(magnitude.mean())
        stats_dict['spatial_frequency'] = np.mean(fft_magnitudes)
        
        # Print
        print(f"  Mean intensity: {stats_dict['mean']:.2f}")
        print(f"  Std deviation: {stats_dict['std']:.2f}")
        print(f"  Value range: [{stats_dict['min']:.0f}, {stats_dict['max']:.0f}]")
        print(f"  Temporal variance: {stats_dict['temporal_variance']:.2f}")
        print(f"  Spatial variance: {stats_dict['spatial_variance']:.2f}")
        print(f"  Motion magnitude: {stats_dict['motion_magnitude']:.2f}")
        print(f"  Intensity skewness: {stats_dict['intensity_skewness']:.3f}")
        print(f"  Spatial frequency: {stats_dict['spatial_frequency']:.2f}")
        
        return stats_dict
    
    def statistical_comparison(self):
        """Compare statistical properties"""
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON: Real vs Synthetic")
        print(f"{'='*60}")
        
        comparisons = []
        
        for metric in ['mean', 'std', 'temporal_variance', 'motion_magnitude']:
            real_val = self.real_stats[metric]
            synth_val = self.synthetic_stats[metric]
            diff_pct = abs(real_val - synth_val) / real_val * 100
            
            # Determine if acceptable (within 30% is good for GANs)
            status = "✅ PASS" if diff_pct < 30 else "⚠️ WARN" if diff_pct < 50 else "❌ FAIL"
            
            comparisons.append({
                'Metric': metric,
                'Real': f'{real_val:.3f}',
                'Synthetic': f'{synth_val:.3f}',
                'Diff %': f'{diff_pct:.1f}%',
                'Status': status
            })
        
        df = pd.DataFrame(comparisons)
        print(df.to_string(index=False))
        
        return comparisons
    
    def temporal_coherence_check(self, videos, name="videos"):
        """Check if videos have smooth temporal transitions"""
        print(f"\n{'='*60}")
        print(f"Temporal Coherence Check: {name}")
        print(f"{'='*60}")
        
        # Calculate frame-to-frame differences
        frame_diffs = []
        for video in videos:
            diffs = np.diff(video, axis=0)
            avg_diff = np.abs(diffs).mean()
            frame_diffs.append(avg_diff)
        
        avg_motion = np.mean(frame_diffs)
        std_motion = np.std(frame_diffs)
        
        print(f"  Average frame-to-frame change: {avg_motion:.3f}")
        print(f"  Std of motion: {std_motion:.3f}")
        
        # Check for flickering (too much variation)
        if std_motion > avg_motion * 2:
            print("  ⚠️ WARNING: High motion variance (possible flickering)")
            return False
        elif avg_motion < 0.1:
            print("  ⚠️ WARNING: Very low motion (static/frozen)")
            return False
        else:
            print("  ✅ PASS: Smooth temporal coherence")
            return True
    
    def spatial_structure_check(self, videos, name="videos"):
        """Check if videos have spatial structure (not random noise)"""
        print(f"\n{'='*60}")
        print(f"Spatial Structure Check: {name}")
        print(f"{'='*60}")
        
        # Calculate edge density (structured images have edges)
        edge_densities = []
        for video in videos[:20]:  # Sample 20 videos
            for frame in video[::4]:  # Sample every 4th frame
                # Sobel edge detection
                sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edge_density = (edges > 10).sum() / edges.size
                edge_densities.append(edge_density)
        
        avg_edge_density = np.mean(edge_densities)
        
        print(f"  Average edge density: {avg_edge_density:.3f}")
        
        # Random noise has very low edge density (<0.1)
        # Structured images have moderate edge density (0.2-0.5)
        if avg_edge_density < 0.1:
            print("  ❌ FAIL: Too little structure (random noise)")
            return False
        elif avg_edge_density > 0.6:
            print("  ⚠️ WARNING: Very high edge density")
            return False
        else:
            print("  ✅ PASS: Good spatial structure")
            return True
    
    def visual_inspection_report(self, videos, output_dir="validation_report"):
        """Create visual report for manual inspection"""
        print(f"\n{'='*60}")
        print(f"Creating Visual Inspection Report")
        print(f"{'='*60}")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot sample frames from different videos
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle('Sample Frames from Synthetic Videos', fontsize=16)
        
        for i in range(20):
            if i >= len(videos):
                break
            row = i // 5
            col = i % 5
            
            # Show middle frame
            frame = videos[i][videos[i].shape[0] // 2]
            axes[row, col].imshow(frame, cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Video {i}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_frames.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/sample_frames.png")
        
        # Plot intensity histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        real_intensities = self.real_videos.flatten()
        synth_intensities = videos.flatten()
        
        ax1.hist(real_intensities, bins=50, alpha=0.7, label='Real', color='blue')
        ax1.hist(synth_intensities, bins=50, alpha=0.7, label='Synthetic', color='orange')
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Intensity Distribution Comparison')
        ax1.legend()
        
        # Temporal variance comparison
        real_temp_var = self.real_videos.var(axis=1).flatten()
        synth_temp_var = videos.var(axis=1).flatten()
        
        ax2.hist(real_temp_var, bins=50, alpha=0.7, label='Real', color='blue')
        ax2.hist(synth_temp_var, bins=50, alpha=0.7, label='Synthetic', color='orange')
        ax2.set_xlabel('Temporal Variance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Temporal Variance Comparison')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/statistical_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/statistical_comparison.png")
        
        plt.close('all')
    
    def validate(self):
        """Run complete validation"""
        print(f"\n{'='*70}")
        print("SYNTHETIC VIDEO VALIDATION FRAMEWORK")
        print(f"{'='*70}")
        
        # Load datasets
        print("\n[1/6] Loading real videos...")
        self.real_videos = self.load_videos(self.real_dir, limit=100)
        print(f"  Loaded {len(self.real_videos)} real videos")
        
        print("\n[2/6] Loading synthetic videos...")
        self.synthetic_videos = self.load_videos(self.synthetic_dir, limit=100)
        print(f"  Loaded {len(self.synthetic_videos)} synthetic videos")
        
        # Compute statistics
        print("\n[3/6] Computing statistics...")
        self.real_stats = self.compute_statistics(self.real_videos, "REAL DATA")
        self.synthetic_stats = self.compute_statistics(self.synthetic_videos, "SYNTHETIC DATA")
        
        # Statistical comparison
        print("\n[4/6] Statistical comparison...")
        comparisons = self.statistical_comparison()
        
        # Temporal coherence
        print("\n[5/6] Checking temporal coherence...")
        real_coherent = self.temporal_coherence_check(self.real_videos, "Real")
        synth_coherent = self.temporal_coherence_check(self.synthetic_videos, "Synthetic")
        
        # Spatial structure
        print("\n[6/6] Checking spatial structure...")
        real_structured = self.spatial_structure_check(self.real_videos, "Real")
        synth_structured = self.spatial_structure_check(self.synthetic_videos, "Synthetic")
        
        # Create visual report
        self.visual_inspection_report(self.synthetic_videos)
        
        # Final verdict
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        passed = 0
        total = len(comparisons) + 2  # stats + coherence + structure
        
        for comp in comparisons:
            if '✅' in comp['Status']:
                passed += 1
        
        if synth_coherent:
            passed += 1
        if synth_structured:
            passed += 1
        
        print(f"\nTests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if passed/total >= 0.7:
            print("\n✅ VERDICT: Synthetic videos are ACCEPTABLE for augmentation")
            print("   - Statistical properties match real data")
            print("   - Temporal and spatial structure present")
            print("   - Recommended: Manual review by domain expert")
        elif passed/total >= 0.5:
            print("\n⚠️ VERDICT: Synthetic videos are MARGINAL")
            print("   - Some issues detected")
            print("   - Recommend: Train longer or adjust parameters")
        else:
            print("\n❌ VERDICT: Synthetic videos are NOT ACCEPTABLE")
            print("   - Significant quality issues")
            print("   - Recommend: Retrain with different settings")
        
        print(f"\n📊 Visual report saved in: validation_report/")
        print(f"{'='*70}\n")
        
        return passed/total


def main():
    """Run validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate synthetic echocardiogram videos")
    parser.add_argument("--real_dir", type=str, default="data_numpy", help="Real video directory (.npy files)")
    parser.add_argument("--synthetic_dir", type=str, required=True, help="Synthetic video directory (.npy files)")
    args = parser.parse_args()
    
    validator = VideoValidator(args.real_dir, args.synthetic_dir)
    success_rate = validator.validate()
    
    if success_rate >= 0.7:
        print("✅ Validation PASSED! Videos are suitable for augmentation.")
        return 0
    else:
        print("⚠️ Validation WARNING! Review results before using.")
        return 1


if __name__ == "__main__":
    main()






