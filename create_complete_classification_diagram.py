"""
Create a comprehensive classification diagram with ALL details
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 5)
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
box_width = 3.5
box_height = 3.5
y_center = 2.5
spacing = 0.6

# Define complete stages with ALL details
stages = [
    {
        'x': 0.3,
        'title': 'INPUT: Video',
        'details': [
            'Format: NumPy array (.npy)',
            'Shape: (32, 64, 64)',
            'Frames: 32 frames uniformly sampled',
            'Resolution: 64×64 pixels',
            'Channels: Grayscale (1 channel)',
            'Normalization: [-1, 1]',
            'Data type: float32'
        ],
        'color': colors['input']
    },
    {
        'x': 0.3 + box_width + spacing,
        'title': '3D CONVOLUTION LAYERS',
        'details': [
            'Layer 1: Conv3d(1→32) kernel=(1,4,4) stride=(1,2,2)',
            '         Output: (32, 32, 32)',
            'Layer 2: Conv3d(32→64) kernel=4 stride=2',
            '         Output: (64, 16, 16)',
            'Layer 3: Conv3d(64→128) kernel=4 stride=2',
            '         Output: (128, 8, 8)',
            'Layer 4: Conv3d(128→256) kernel=4 stride=2',
            '         Output: (256, 4, 4, 4)',
            'Activation: ReLU, BatchNorm after each'
        ],
        'color': colors['conv']
    },
    {
        'x': 0.3 + 2*(box_width + spacing),
        'title': 'GLOBAL POOLING',
        'details': [
            'Type: Adaptive Average Pooling',
            'Input: (256, 4, 4, 4)',
            'Output: (256, 1, 1, 1)',
            'Operation: Average across spatial & temporal',
            'Result: 256-dimensional feature vector',
            'Purpose: Dimension reduction & summarization'
        ],
        'color': colors['pool']
    },
    {
        'x': 0.3 + 3*(box_width + spacing),
        'title': 'FULLY CONNECTED LAYERS',
        'details': [
            'FC Layer 1: Linear(256 → 128)',
            '           Activation: ReLU',
            '           Dropout: 0.5',
            'FC Layer 2: Linear(128 → 5)',
            '           Output: 5 logits (one per age bin)',
            'Total Parameters: ~50K',
            'Optimizer: Adam (lr=1e-4)'
        ],
        'color': colors['fc']
    },
    {
        'x': 0.3 + 4*(box_width + spacing),
        'title': 'OUTPUT: Age Classification',
        'details': [
            'Task: Multi-class classification',
            'Classes: 5 age bins',
            '  0-1 years',
            '  2-5 years',
            '  6-10 years',
            '  11-15 years',
            '  16-18 years',
            'Loss: CrossEntropyLoss',
            'Prediction: Argmax of 5 logits'
        ],
        'color': colors['output']
    }
]

# Draw boxes with all details
for stage in stages:
    # Draw main box
    box = Rectangle((stage['x'], y_center - box_height/2), 
                   box_width, box_height,
                   facecolor=stage['color'], alpha=0.12,
                   edgecolor=stage['color'], linewidth=3)
    ax.add_patch(box)
    
    # Title
    ax.text(stage['x'] + box_width/2, y_center + box_height/2 - 0.25,
           stage['title'],
           ha='center', va='top',
           fontsize=13, fontweight='bold',
           color=stage['color'])
    
    # Details (all lines)
    y_start = y_center + box_height/2 - 0.6
    line_height = 0.25
    for i, detail in enumerate(stage['details']):
        ax.text(stage['x'] + box_width/2, y_start - i * line_height,
               detail,
               ha='center', va='top',
               fontsize=9,
               color='#2C3E50',
               family='monospace')

# Draw arrows between boxes
for i in range(len(stages) - 1):
    x_from = stages[i]['x'] + box_width
    x_to = stages[i+1]['x']
    arrow = FancyArrowPatch((x_from, y_center), (x_to, y_center),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3.5, color='#34495E')
    ax.add_patch(arrow)

# Main title
ax.text(10, 4.5, '3D CNN Classifier - Complete Architecture Details',
       ha='center', fontsize=20, fontweight='bold', color='#2C3E50')

# Add summary box at bottom
summary_box = Rectangle((1, 0.1), 18, 0.6,
                       facecolor='#ECF0F1', alpha=0.8,
                       edgecolor='#BDC3C7', linewidth=2)
ax.add_patch(summary_box)
summary_text = (
    'Model Summary: 3D CNN | Input: (32, 64, 64) | '
    '4 Conv Layers → Global Pooling → 2 FC Layers | '
    'Output: 5 Age Classes | Training: 10 epochs, Adam optimizer, Batch size: 8'
)
ax.text(10, 0.4, summary_text,
       ha='center', va='center',
       fontsize=10, fontweight='bold',
       color='#2C3E50')

plt.tight_layout()
plt.savefig('classification_complete_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved complete classification diagram to: classification_complete_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created COMPLETE classification diagram with ALL details!")
print("="*80)
print("\nIncludes:")
print("  - Input specifications (shape, format, normalization)")
print("  - All 4 convolution layers with exact parameters")
print("  - Pooling operation details")
print("  - FC layers with dimensions and activations")
print("  - Output specifications (age bins, loss function)")
print("  - Model summary at bottom")







