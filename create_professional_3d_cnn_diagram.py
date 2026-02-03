"""
Create a clean, professional diagram for 3D CNN Classifier
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# Create figure with tight layout
fig, ax = plt.subplots(1, 1, figsize=(16, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')

# Professional colors
colors = {
    'input': '#4A90E2',
    'conv': '#50C878',
    'pool': '#FF6B6B',
    'fc': '#9B59B6',
    'output': '#F39C12'
}

# Box dimensions
box_width = 2.5
box_height = 2.2
y_center = 2
x_start = 1

# Stage 1: Video Input
x1 = x_start
# Draw clean video frames representation (3x3 grid)
frame_size = 0.15
for i in range(3):
    for j in range(3):
        x_pos = x1 + 0.3 + j * frame_size * 1.2
        y_pos = y_center + 0.4 - i * frame_size * 1.2
        rect = Rectangle((x_pos, y_pos), frame_size, frame_size,
                        facecolor='#E8E8E8',
                        edgecolor='#333333',
                        linewidth=0.8)
        ax.add_patch(rect)

# Box
box1 = Rectangle((x1, y_center - box_height/2), box_width, box_height,
                facecolor=colors['input'], alpha=0.2,
                edgecolor=colors['input'], linewidth=2.5)
ax.add_patch(box1)
ax.text(x1 + box_width/2, y_center + 0.7, 'Video Input',
        ha='center', fontsize=13, fontweight='bold', color=colors['input'])
ax.text(x1 + box_width/2, y_center, '32 frames\n64×64',
        ha='center', fontsize=10, color='#333333')

# Stage 2: 3D Convolution
x2 = x1 + box_width + 0.8
# Draw convolution operation (clean representation)
conv_size = 0.2
# Feature map
fm = Rectangle((x2 + 0.4, y_center + 0.2), conv_size * 1.5, conv_size * 1.5,
              facecolor=colors['conv'], alpha=0.3,
              edgecolor=colors['conv'], linewidth=2)
ax.add_patch(fm)
# Kernel
kernel = Rectangle((x2 + 0.6, y_center + 0.3), conv_size * 0.6, conv_size * 0.6,
                   facecolor='#FFA500', alpha=0.7,
                   edgecolor='#FF8C00', linewidth=1.5)
ax.add_patch(kernel)

box2 = Rectangle((x2, y_center - box_height/2), box_width, box_height,
                facecolor=colors['conv'], alpha=0.2,
                edgecolor=colors['conv'], linewidth=2.5)
ax.add_patch(box2)
ax.text(x2 + box_width/2, y_center + 0.7, '3D Convolution',
        ha='center', fontsize=13, fontweight='bold', color=colors['conv'])
ax.text(x2 + box_width/2, y_center, '4 layers\nFeature extraction',
        ha='center', fontsize=10, color='#333333')

# Stage 3: Pooling
x3 = x2 + box_width + 0.8
# Draw pooling (clean before/after)
before = Rectangle((x3 + 0.3, y_center + 0.1), 0.4, 0.4,
                  facecolor=colors['pool'], alpha=0.4,
                  edgecolor=colors['pool'], linewidth=2)
ax.add_patch(before)
after = Rectangle((x3 + 1.1, y_center + 0.2), 0.2, 0.2,
                 facecolor=colors['pool'], alpha=0.8,
                 edgecolor=colors['pool'], linewidth=2)
ax.add_patch(after)
# Arrow
arrow_pool = FancyArrowPatch((x3 + 0.7, y_center + 0.3), (x3 + 1.1, y_center + 0.3),
                             arrowstyle='->', mutation_scale=15,
                             linewidth=2, color=colors['pool'])
ax.add_patch(arrow_pool)

box3 = Rectangle((x3, y_center - box_height/2), box_width, box_height,
                facecolor=colors['pool'], alpha=0.2,
                edgecolor=colors['pool'], linewidth=2.5)
ax.add_patch(box3)
ax.text(x3 + box_width/2, y_center + 0.7, 'Pooling',
        ha='center', fontsize=13, fontweight='bold', color=colors['pool'])
ax.text(x3 + box_width/2, y_center, 'Global pooling\nDimension reduction',
        ha='center', fontsize=10, color='#333333')

# Stage 4: FC Layers
x4 = x3 + box_width + 0.8
# Draw clean neural network (simplified)
node_size = 0.08
# Input nodes
for i in range(3):
    y_pos = y_center + 0.3 - i * 0.25
    circle = Circle((x4 + 0.5, y_pos), node_size,
                   facecolor=colors['fc'],
                   edgecolor='#333333', linewidth=1.5)
    ax.add_patch(circle)
# Output nodes
for i in range(3):
    y_pos = y_center + 0.3 - i * 0.25
    circle = Circle((x4 + 1.5, y_pos), node_size,
                   facecolor=colors['fc'],
                   edgecolor='#333333', linewidth=1.5)
    ax.add_patch(circle)
# Connections (subtle)
for i in range(3):
    y1 = y_center + 0.3 - i * 0.25
    for j in range(3):
        y2 = y_center + 0.3 - j * 0.25
        line = plt.Line2D([x4 + 0.58, x4 + 1.42], [y1, y2],
                          linewidth=0.8, color=colors['fc'], alpha=0.2)
        ax.add_line(line)

box4 = Rectangle((x4, y_center - box_height/2), box_width, box_height,
                facecolor=colors['fc'], alpha=0.2,
                edgecolor=colors['fc'], linewidth=2.5)
ax.add_patch(box4)
ax.text(x4 + box_width/2, y_center + 0.7, 'FC Layers',
        ha='center', fontsize=13, fontweight='bold', color=colors['fc'])
ax.text(x4 + box_width/2, y_center, '256→128→5\nClassification',
        ha='center', fontsize=10, color='#333333')

# Stage 5: Age Class
x5 = x4 + box_width + 0.8
# Draw age bins (clean vertical bars)
bin_width = 0.35
bin_height = 0.25
ages = ['0-1', '2-5', '6-10', '11-15', '16-18']
for i, age in enumerate(ages):
    y_pos = y_center + 0.8 - i * bin_height
    if i == 2:  # Highlight prediction
        rect = Rectangle((x5 + 0.5, y_pos - bin_height/2), bin_width, bin_height,
                        facecolor=colors['output'],
                        edgecolor='#333333', linewidth=2.5)
    else:
        rect = Rectangle((x5 + 0.5, y_pos - bin_height/2), bin_width, bin_height,
                        facecolor='#E8E8E8',
                        edgecolor='#CCCCCC', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x5 + 0.675, y_pos, age,
           ha='center', va='center', fontsize=9, fontweight='bold')

box5 = Rectangle((x5, y_center - box_height/2), box_width, box_height,
                facecolor=colors['output'], alpha=0.2,
                edgecolor=colors['output'], linewidth=2.5)
ax.add_patch(box5)
ax.text(x5 + box_width/2, y_center + 0.7, 'Age Class',
        ha='center', fontsize=13, fontweight='bold', color=colors['output'])
ax.text(x5 + box_width/2, y_center - 0.8, '5 age bins\nPrediction',
        ha='center', fontsize=10, color='#333333')

# Add clean arrows between stages
for i, x_from in enumerate([x1 + box_width, x2 + box_width, x3 + box_width, x4 + box_width]):
    x_to = [x2, x3, x4, x5][i]
    arrow = FancyArrowPatch((x_from, y_center), (x_to, y_center),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='#666666')
    ax.add_patch(arrow)

# Title
ax.text(8, 3.5, '3D CNN Classifier Architecture', 
        ha='center', fontsize=16, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig('3d_cnn_diagram_professional.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved professional diagram to: 3d_cnn_diagram_professional.png")
plt.close()

print("\n" + "="*70)
print("Created clean, professional diagram!")
print("="*70)
print("\nFeatures:")
print("  - Clean, modern design")
print("  - Proper spacing (no clutter)")
print("  - Professional visualizations")
print("  - No overlapping elements")
print("  - Suitable for academic presentation")







