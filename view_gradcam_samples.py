"""
Interactive viewer for GradCAM results
Shows multiple sample visualizations in a grid
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np


def show_gradcam_samples(num_samples=6):
    """Display multiple GradCAM visualizations in a grid"""
    viz_dir = Path("gradcam_results/visualizations")
    
    # Get all visualization files
    viz_files = sorted(list(viz_dir.glob("*.png")))[:num_samples]
    
    if not viz_files:
        print("No visualizations found!")
        return
    
    # Create grid
    cols = 3
    rows = (len(viz_files) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, viz_path in enumerate(viz_files):
        row = idx // cols
        col = idx % cols
        
        img = mpimg.imread(viz_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(Path(viz_path).stem.replace('_gradcam', ''), fontsize=10)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(len(viz_files), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"GradCAM Heatmaps for Synthetic Echocardiogram Videos\n(Showing {len(viz_files)} of 100)", 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig("gradcam_results/gradcam_samples_grid.png", dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization: gradcam_results/gradcam_samples_grid.png")
    plt.show()


if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    show_gradcam_samples(num_samples)


