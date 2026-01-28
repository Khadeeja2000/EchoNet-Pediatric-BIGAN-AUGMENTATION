"""
Create an interactive, visual diagram for 3D CNN Classifier
with actual visual representations at each stage
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec

# Create figure with better layout
fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

# Colors
colors = {
    'input': '#E3F2FD',
    'conv': '#BBDEFB',
    'pool': '#90CAF9',
    'fc': '#64B5F6',
    'output': '#42A5F5'
}

# Stage 1: Video Input
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Draw video frames representation
for i in range(3):
    for j in range(3):
        x = 0.1 + j * 0.3
        y = 0.6 - i * 0.3
        rect = Rectangle((x, y), 0.25, 0.25, 
                        facecolor='lightgray', 
                        edgecolor='black', 
                        linewidth=1)
        ax1.add_patch(rect)
        # Add frame number
        ax1.text(x + 0.125, y + 0.125, f'{i*3+j+1}', 
                ha='center', va='center', fontsize=8)

ax1.text(0.5, 0.15, 'Video Input\n(32 frames)', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=colors['input'], edgecolor='black', linewidth=2))

# Stage 2: 3D Convolution
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Draw convolution kernel visualization
kernel_size = 0.15
for i in range(2):
    for j in range(2):
        x = 0.2 + j * 0.3
        y = 0.6 - i * 0.3
        # Draw kernel
        kernel = Rectangle((x, y), kernel_size, kernel_size,
                          facecolor='orange', 
                          edgecolor='black', 
                          linewidth=1.5,
                          alpha=0.7)
        ax2.add_patch(kernel)
        # Draw input feature map
        feature_map = Rectangle((x + 0.2, y), kernel_size*1.5, kernel_size*1.5,
                               facecolor='lightblue',
                               edgecolor='blue',
                               linewidth=1)
        ax2.add_patch(feature_map)

# Draw arrow showing convolution operation
arrow1 = FancyArrowPatch((0.5, 0.3), (0.5, 0.2),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='red')
ax2.add_patch(arrow1)

ax2.text(0.5, 0.05, '3D Convolution\n(4 layers)', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=colors['conv'], edgecolor='black', linewidth=2))

# Stage 3: Pooling
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Draw pooling operation
pool_size = 0.2
# Before pooling (larger)
before = Rectangle((0.15, 0.5), pool_size*2, pool_size*2,
                  facecolor='lightblue',
                  edgecolor='blue',
                  linewidth=2)
ax3.add_patch(before)
ax3.text(0.25, 0.6, 'Before', fontsize=9, ha='center')

# After pooling (smaller)
after = Rectangle((0.65, 0.55), pool_size, pool_size,
                 facecolor='darkblue',
                 edgecolor='black',
                 linewidth=2)
ax3.add_patch(after)
ax3.text(0.7, 0.6, 'After', fontsize=9, ha='center')

# Arrow showing pooling
arrow2 = FancyArrowPatch((0.35, 0.6), (0.65, 0.6),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='red')
ax3.add_patch(arrow2)

ax3.text(0.5, 0.05, 'Global Pooling', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=colors['pool'], edgecolor='black', linewidth=2))

# Stage 4: FC Layers
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Draw neural network connections
# Input layer (left)
for i in range(3):
    y_pos = 0.7 - i * 0.2
    circle1 = Circle((0.2, y_pos), 0.05, 
                    facecolor='lightgreen',
                    edgecolor='black',
                    linewidth=1.5)
    ax4.add_patch(circle1)

# Output layer (right)
for i in range(3):
    y_pos = 0.7 - i * 0.2
    circle2 = Circle((0.8, y_pos), 0.05,
                    facecolor='lightcoral',
                    edgecolor='black',
                    linewidth=1.5)
    ax4.add_patch(circle2)
    # Draw connections
    for j in range(3):
        y_pos2 = 0.7 - j * 0.2
        line = plt.Line2D([0.25, 0.75], [y_pos, y_pos2],
                         linewidth=0.5, color='gray', alpha=0.3)
        ax4.add_line(line)

ax4.text(0.5, 0.05, 'FC Layers\n(256→128→5)', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=colors['fc'], edgecolor='black', linewidth=2))

# Stage 5: Age Class Output
ax5 = fig.add_subplot(gs[0, 4])
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

# Draw 5 age bins as boxes
age_bins = ['0-1', '2-5', '6-10', '11-15', '16-18']
bin_height = 0.12
for i, age in enumerate(age_bins):
    y_pos = 0.85 - i * bin_height
    if i == 2:  # Highlight one (predicted)
        rect = Rectangle((0.2, y_pos - bin_height/2), 0.6, bin_height,
                        facecolor='yellow',
                        edgecolor='red',
                        linewidth=2.5)
    else:
        rect = Rectangle((0.2, y_pos - bin_height/2), 0.6, bin_height,
                        facecolor='lightgray',
                        edgecolor='black',
                        linewidth=1)
    ax5.add_patch(rect)
    ax5.text(0.5, y_pos, age, ha='center', va='center', 
            fontsize=10, fontweight='bold')

ax5.text(0.5, 0.05, 'Age Class\n(5 bins)', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=colors['output'], edgecolor='black', linewidth=2))

# Add arrows between stages
for i in range(4):
    ax_arrow = fig.add_subplot(gs[0, i+1])
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')
    arrow = FancyArrowPatch((0.05, 0.5), (0.95, 0.5),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=3, color='darkblue')
    ax_arrow.add_patch(arrow)

# Add main title
fig.suptitle('3D CNN Classifier Architecture - Visual Flow', 
             fontsize=18, fontweight='bold', y=0.95)

# Add bottom row with descriptions
desc_ax = fig.add_subplot(gs[1, :])
desc_ax.axis('off')
descriptions = [
    'Video frames\n(32 frames × 64×64)',
    'Feature extraction\n(Convolution filters)',
    'Dimension reduction\n(Summarize features)',
    'Classification\n(Neural network)',
    'Age prediction\n(5 categories)'
]

for i, desc in enumerate(descriptions):
    x_pos = 0.1 + i * 0.2
    desc_ax.text(x_pos, 0.5, desc,
                ha='center', va='center',
                fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('3d_cnn_diagram_interactive.png', dpi=300, bbox_inches='tight')
print("✓ Saved interactive diagram to: 3d_cnn_diagram_interactive.png")
plt.close()

print("\n" + "="*70)
print("Created interactive diagram with visual representations!")
print("="*70)
print("\nEach stage shows:")
print("  1. Video Input: Grid of video frames")
print("  2. 3D Conv: Convolution kernels operating on features")
print("  3. Pooling: Size reduction visualization")
print("  4. FC Layers: Neural network connections")
print("  5. Age Class: 5 age bins with highlighted prediction")







