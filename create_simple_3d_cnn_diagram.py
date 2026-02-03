"""
Create a simple, clean diagram with just boxes and detailed step names
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 3))
ax.set_xlim(0, 16)
ax.set_ylim(0, 3)
ax.axis('off')

# Professional color scheme
colors = {
    'input': '#3498DB',
    'conv': '#2ECC71',
    'pool': '#E74C3C',
    'fc': '#9B59B6',
    'output': '#F39C12'
}

# Box dimensions
box_width = 2.8
box_height = 2
y_center = 1.5
spacing = 0.5

# Define stages with detailed names
stages = [
    {
        'x': 0.5,
        'title': 'Video Input',
        'details': '32 frames\n64×64 pixels\nGrayscale',
        'color': colors['input']
    },
    {
        'x': 0.5 + box_width + spacing,
        'title': '3D Convolution',
        'details': '4 convolution layers\nFeature extraction\nSpatial + Temporal',
        'color': colors['conv']
    },
    {
        'x': 0.5 + 2*(box_width + spacing),
        'title': 'Global Pooling',
        'details': 'Dimension reduction\nSummarize features\n256-dim vector',
        'color': colors['pool']
    },
    {
        'x': 0.5 + 3*(box_width + spacing),
        'title': 'Fully Connected Layers',
        'details': '256 → 128 neurons\n128 → 5 neurons\nClassification',
        'color': colors['fc']
    },
    {
        'x': 0.5 + 4*(box_width + spacing),
        'title': 'Age Classification',
        'details': '5 age bins\n0-1, 2-5, 6-10\n11-15, 16-18 years',
        'color': colors['output']
    }
]

# Draw boxes
for stage in stages:
    # Draw box
    box = Rectangle((stage['x'], y_center - box_height/2), 
                   box_width, box_height,
                   facecolor=stage['color'], alpha=0.15,
                   edgecolor=stage['color'], linewidth=2.5)
    ax.add_patch(box)
    
    # Title
    ax.text(stage['x'] + box_width/2, y_center + 0.6,
           stage['title'],
           ha='center', va='center',
           fontsize=14, fontweight='bold',
           color=stage['color'])
    
    # Details
    ax.text(stage['x'] + box_width/2, y_center - 0.2,
           stage['details'],
           ha='center', va='center',
           fontsize=11,
           color='#2C3E50',
           linespacing=1.4)

# Draw arrows between boxes
for i in range(len(stages) - 1):
    x_from = stages[i]['x'] + box_width
    x_to = stages[i+1]['x']
    arrow = FancyArrowPatch((x_from, y_center), (x_to, y_center),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=3, color='#34495E')
    ax.add_patch(arrow)

# Title
ax.text(8, 2.6, '3D CNN Classifier Architecture',
       ha='center', fontsize=18, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig('3d_cnn_diagram_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved simple diagram to: 3d_cnn_diagram_simple.png")
plt.close()

print("\n" + "="*70)
print("Created simple diagram with boxes and detailed step names!")
print("="*70)







