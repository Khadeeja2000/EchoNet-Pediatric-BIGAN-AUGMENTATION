"""
Display GradCAM results for synthetic videos
"""
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def show_gradcam_visualization(viz_path):
    """Display a GradCAM visualization image"""
    if os.path.exists(viz_path):
        img = plt.imread(viz_path)
        plt.figure(figsize=(16, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"GradCAM Visualization: {Path(viz_path).name}", fontsize=14)
        plt.tight_layout()
        plt.show()
        print(f"Displayed: {viz_path}")
    else:
        print(f"File not found: {viz_path}")


def list_gradcam_results(results_dir="gradcam_results"):
    """List all GradCAM results"""
    results_dir = Path(results_dir)
    
    visualizations = sorted(list((results_dir / "visualizations").glob("*.png")))
    overlays = sorted(list((results_dir / "overlays").glob("*.mp4")))
    
    print("=" * 60)
    print("GRADCAM RESULTS FOR SYNTHETIC VIDEOS")
    print("=" * 60)
    print(f"\nVisualizations (PNG): {len(visualizations)}")
    print(f"Overlay Videos (MP4): {len(overlays)}")
    
    if visualizations:
        print("\nSample visualizations:")
        for viz in visualizations[:5]:
            print(f"  - {viz.name}")
    
    if overlays:
        print("\nSample overlay videos:")
        for ovl in overlays[:5]:
            print(f"  - {ovl.name}")
    
    print("\n" + "=" * 60)
    print("WHAT THE HEATMAPS SHOW:")
    print("=" * 60)
    print("""
The GradCAM heatmaps highlight regions in the synthetic videos that are
most important for the discriminator's decision-making:

- RED/YELLOW areas: High importance (strong activation)
- BLUE areas: Low importance (weak activation)

This helps understand:
1. Which anatomical regions the model focuses on
2. Whether synthetic videos have realistic cardiac structures
3. Spatial patterns in the generated videos
4. Potential artifacts or unrealistic features

The overlays blend the heatmap with the original video for easy viewing.
    """)
    
    return visualizations, overlays


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Show specific visualization
        show_gradcam_visualization(sys.argv[1])
    else:
        # List all results
        visualizations, overlays = list_gradcam_results()
        
        if visualizations:
            print(f"\nTo view a visualization, run:")
            print(f"  python3 show_gradcam_results.py {visualizations[0]}")


