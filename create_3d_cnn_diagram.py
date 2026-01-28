"""
Create a simple flowchart diagram for 3D CNN Classifier
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

# Define box properties
box_width = 2.2
box_height = 1.8
y_center = 2.5
spacing = 2.8

# Colors
colors = {
    'input': '#E3F2FD',  # Light blue
    'conv': '#BBDEFB',   # Medium blue
    'pool': '#90CAF9',   # Darker blue
    'fc': '#64B5F6',     # Blue
    'output': '#42A5F5'  # Dark blue
}

# Boxes - simplified text with only most important info
boxes = [
    {'x': 1, 'label': 'Video\nInput', 'color': colors['input']},
    {'x': 4.2, 'label': '3D Conv\n(4 layers)', 'color': colors['conv']},
    {'x': 7.4, 'label': 'Pooling', 'color': colors['pool']},
    {'x': 10.6, 'label': 'FC Layers', 'color': colors['fc']},
    {'x': 13.8, 'label': 'Age\nClass', 'color': colors['output']}
]

# Draw boxes
for i, box in enumerate(boxes):
    # Draw rounded rectangle
    rect = mpatches.FancyBboxPatch(
        (box['x'], y_center - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=box['color'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(rect)
    
    # Add text
    ax.text(box['x'] + box_width/2, y_center, box['label'],
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw arrow (except for last box)
    if i < len(boxes) - 1:
        arrow = mpatches.FancyArrowPatch(
            (box['x'] + box_width, y_center),
            (boxes[i+1]['x'], y_center),
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='black'
        )
        ax.add_patch(arrow)

# Title
ax.text(8, 4.2, '3D CNN Classifier Architecture', 
        ha='center', fontsize=16, fontweight='bold')

# Save
plt.tight_layout()
plt.savefig('3d_cnn_diagram.png', dpi=300, bbox_inches='tight')
print("✓ Saved diagram to: 3d_cnn_diagram.png")
plt.close()

# Also create a simpler text version
print("\n" + "="*60)
print("Simple Text Diagram:")
print("="*60)
print("""
    Video Input      →      3D Conv      →      Pooling      →      FC Layers      →      Age Class
   (32×64×64)              (4 layers)          (Global)            (256→128→5)          (5 bins)
""")

