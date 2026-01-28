"""
Create diagram with detailed headings only (no details inside boxes)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.set_xlim(0, 20)
ax.set_ylim(0, 4)
ax.axis('off')

# Color scheme
colors = {
    'input': '#3498DB',
    'conv': '#2ECC71',
    'pool': '#E74C3C',
    'fc': '#9B59B6',
    'output': '#F39C12'
}

# Box dimensions
box_width = 3.6
box_height = 2.5
y_center = 2
spacing = 0.5

# Define stages with detailed headings (all info in title)
stages = [
    {
        'x': 0.2,
        'heading': 'INPUT: Video\n(32 frames × 64×64 pixels, Grayscale, Normalized [-1,1], NumPy array)',
        'color': colors['input']
    },
    {
        'x': 0.2 + box_width + spacing,
        'heading': '3D CONVOLUTION: 4 Layers\n(Conv3d: 1→32→64→128→256, Kernel=4, Stride=2, ReLU+BatchNorm)',
        'color': colors['conv']
    },
    {
        'x': 0.2 + 2*(box_width + spacing),
        'heading': 'GLOBAL POOLING\n(AdaptiveAvgPool3d: (256,4,4,4) → (256,1,1,1), 256-dim feature vector)',
        'color': colors['pool']
    },
    {
        'x': 0.2 + 3*(box_width + spacing),
        'heading': 'FULLY CONNECTED LAYERS\n(FC1: 256→128 ReLU+Dropout0.5, FC2: 128→5, Adam optimizer lr=1e-4)',
        'color': colors['fc']
    },
    {
        'x': 0.2 + 4*(box_width + spacing),
        'heading': 'OUTPUT: Age Classification\n(5 classes: 0-1, 2-5, 6-10, 11-15, 16-18 years, CrossEntropyLoss)',
        'color': colors['output']
    }
]

# Draw boxes with detailed headings only
for stage in stages:
    # Draw box
    box = Rectangle((stage['x'], y_center - box_height/2), 
                   box_width, box_height,
                   facecolor=stage['color'], alpha=0.15,
                   edgecolor=stage['color'], linewidth=3)
    ax.add_patch(box)
    
    # Detailed heading (centered in box)
    ax.text(stage['x'] + box_width/2, y_center,
           stage['heading'],
           ha='center', va='center',
           fontsize=10.5,
           fontweight='bold',
           color='#2C3E50',
           linespacing=1.5)

# Draw arrows between boxes
for i in range(len(stages) - 1):
    x_from = stages[i]['x'] + box_width
    x_to = stages[i+1]['x']
    arrow = FancyArrowPatch((x_from, y_center), (x_to, y_center),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3.5, color='#34495E')
    ax.add_patch(arrow)

# Main title
ax.text(10, 3.5, '3D CNN Classifier Architecture - Complete Flow',
       ha='center', fontsize=18, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig('classification_headings_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved diagram with detailed headings to: classification_headings_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created diagram with detailed headings only!")
print("="*80)
print("\nEach box heading contains:")
print("  - All technical specifications")
print("  - Complete model details")
print("  - No bullet points inside boxes")
print("  - Everything in the heading itself")







