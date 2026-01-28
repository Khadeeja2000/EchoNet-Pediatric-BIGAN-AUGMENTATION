"""
Create clean diagram with concise but detailed headings
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(18, 3.5))
ax.set_xlim(0, 18)
ax.set_ylim(0, 3.5)
ax.axis('off')

# Color scheme
colors = {
    'input': '#3498DB',
    'conv': '#2ECC71',
    'pool': '#E74C3C',
    'fc': '#9B59B6',
    'output': '#F39C12'
}

# Box dimensions - larger for readability
box_width = 3.2
box_height = 2
y_center = 1.75
spacing = 0.7

# Define stages with cleaner, concise headings
stages = [
    {
        'x': 0.3,
        'title': 'Video Input',
        'subtitle': '32 frames × 64×64\nGrayscale, Normalized',
        'color': colors['input']
    },
    {
        'x': 0.3 + box_width + spacing,
        'title': '3D Convolution',
        'subtitle': '4 layers: 1→32→64→128→256\nKernel=4, Stride=2',
        'color': colors['conv']
    },
    {
        'x': 0.3 + 2*(box_width + spacing),
        'title': 'Global Pooling',
        'subtitle': '(256,4,4,4) → (256,1,1,1)\n256-dim vector',
        'color': colors['pool']
    },
    {
        'x': 0.3 + 3*(box_width + spacing),
        'title': 'FC Layers',
        'subtitle': '256→128 (ReLU+Dropout)\n128→5 (Adam lr=1e-4)',
        'color': colors['fc']
    },
    {
        'x': 0.3 + 4*(box_width + spacing),
        'title': 'Age Classification',
        'subtitle': '5 classes: 0-1, 2-5, 6-10\n11-15, 16-18 years',
        'color': colors['output']
    }
]

# Draw boxes with clean headings
for stage in stages:
    # Draw box
    box = Rectangle((stage['x'], y_center - box_height/2), 
                   box_width, box_height,
                   facecolor=stage['color'], alpha=0.12,
                   edgecolor=stage['color'], linewidth=2.5)
    ax.add_patch(box)
    
    # Title (larger, bold)
    ax.text(stage['x'] + box_width/2, y_center + 0.4,
           stage['title'],
           ha='center', va='center',
           fontsize=13, fontweight='bold',
           color=stage['color'])
    
    # Subtitle (smaller, regular)
    ax.text(stage['x'] + box_width/2, y_center - 0.3,
           stage['subtitle'],
           ha='center', va='center',
           fontsize=10,
           color='#2C3E50',
           linespacing=1.3)

# Draw arrows between boxes
for i in range(len(stages) - 1):
    x_from = stages[i]['x'] + box_width
    x_to = stages[i+1]['x']
    arrow = FancyArrowPatch((x_from, y_center), (x_to, y_center),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='#34495E')
    ax.add_patch(arrow)

# Main title
ax.text(9, 3, '3D CNN Classifier Architecture',
       ha='center', fontsize=16, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig('classification_clean_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved clean diagram to: classification_clean_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created clean diagram with concise headings!")
print("="*80)
print("\nFeatures:")
print("  - Clean, uncluttered layout")
print("  - Title + Subtitle format (easier to read)")
print("  - Better spacing between boxes")
print("  - All essential details included")
print("  - Professional appearance")







