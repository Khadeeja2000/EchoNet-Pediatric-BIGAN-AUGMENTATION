"""
Create tree flow diagram with boxes and arrows for classification
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
colors = {
    'main': '#2C3E50',
    'model': '#3498DB',
    'input': '#E74C3C',
    'conv': '#2ECC71',
    'pool': '#F39C12',
    'fc': '#9B59B6',
    'output': '#E67E22',
    'train': '#16A085',
    'result': '#C0392B'
}

def draw_box(x, y, width, height, text_lines, color, fontsize=10):
    """Draw a box with text"""
    # Box
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color, alpha=0.2,
                         edgecolor=color, linewidth=2.5)
    ax.add_patch(box)
    
    # Text
    y_start = y + height - 0.15
    line_height = 0.3
    for i, line in enumerate(text_lines):
        ax.text(x + width/2, y_start - i * line_height,
               line, ha='center', va='top',
               fontsize=fontsize, fontweight='bold' if i == 0 else 'normal',
               color='#2C3E50')

def draw_arrow(x1, y1, x2, y2, color='#34495E', width=2):
    """Draw arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=width, color=color)
    ax.add_patch(arrow)

# Top: Classification Test
draw_box(7, 12.5, 6, 1, 
         ['CLASSIFICATION TEST', 'Purpose: Test synthetic videos', 'Result: -2.17% (doesn\'t improve)'],
         colors['main'], fontsize=11)

# Model Box
draw_box(7, 10.5, 6, 1.2,
         ['MODEL: 3D CNN CLASSIFIER', 'Processes spatial + temporal', 'Standard for video classification'],
         colors['model'], fontsize=11)

# Arrow from top to model
draw_arrow(10, 12.5, 10, 11.7)

# Architecture boxes (horizontal flow)
# Input
draw_box(1, 8, 2.5, 1,
         ['INPUT: Video', '32 frames × 64×64', 'Grayscale, normalized'],
         colors['input'], fontsize=9)

# Conv Layers
draw_box(4.5, 8, 3, 1,
         ['3D CONVOLUTION', '4 layers: 1→32→64→128→256', 'Kernel=4, Stride=2'],
         colors['conv'], fontsize=9)

# Pooling
draw_box(8.5, 8, 2.5, 1,
         ['GLOBAL POOLING', '(256,4,4,4) → (256,1,1,1)', '256-dim vector'],
         colors['pool'], fontsize=9)

# FC Layers
draw_box(12, 8, 2.5, 1,
         ['FC LAYERS', '256→128 (ReLU+Dropout)', '128→5 (logits)'],
         colors['fc'], fontsize=9)

# Output
draw_box(15.5, 8, 2.5, 1,
         ['OUTPUT', '5 age bins', 'CrossEntropyLoss'],
         colors['output'], fontsize=9)

# Arrows between architecture boxes
draw_arrow(3.5, 8.5, 4.5, 8.5)
draw_arrow(7.5, 8.5, 8.5, 8.5)
draw_arrow(11, 8.5, 12, 8.5)
draw_arrow(14.5, 8.5, 15.5, 8.5)

# Arrow from model to architecture
draw_arrow(10, 10.5, 10, 9)

# Training & Data section
draw_box(2, 5.5, 4, 1.5,
         ['TRAINING CONFIG', 'Epochs: 10', 'Adam (lr=1e-4), Batch: 8'],
         colors['train'], fontsize=9)

draw_box(8, 5.5, 4, 1.5,
         ['DATA SPLIT', 'Train: 1,290 (80%)', 'Val: 323 (20%), Synth: 100'],
         colors['train'], fontsize=9)

draw_box(14, 5.5, 4, 1.5,
         ['EXPERIMENTS', 'Baseline: 39.63%', 'Augmented: 37.46%'],
         colors['result'], fontsize=9)

# Arrows from architecture to training
draw_arrow(2.25, 8, 4, 7)
draw_arrow(10, 8, 10, 7)
draw_arrow(17.75, 8, 16, 7)

# Title
ax.text(10, 13.5, '3D CNN Classifier - Complete Flow',
       ha='center', fontsize=18, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig('classification_tree_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved tree flow diagram with boxes and arrows to: classification_tree_flow_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created tree flow diagram with boxes and arrows!")
print("="*80)







