"""
Create C3D-GAN architecture diagram similar to BiGAN diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_features = '#E8E8FF'  # Light purple
color_data = '#E8FFE8'      # Light green
color_generator = '#FFE8E8'  # Light red
color_discriminator = '#FFFFE8'  # Light yellow

# Draw Features Domain box
features_box = Rectangle((0.5, 6), 7, 3.5,
                        facecolor=color_features, alpha=0.3,
                        edgecolor='purple', linewidth=2)
ax.add_patch(features_box)
ax.text(4, 9.2, 'features', fontsize=14, fontweight='bold', 
        ha='center', color='purple')

# Draw Data Domain box
data_box = Rectangle((0.5, 0.5), 7, 3.5,
                    facecolor=color_data, alpha=0.3,
                    edgecolor='green', linewidth=2)
ax.add_patch(data_box)
ax.text(4, 3.7, 'data', fontsize=14, fontweight='bold',
        ha='center', color='green')

# Features Domain - Random Noise z
z_circle = Circle((2, 8), 0.4, facecolor='white', 
                  edgecolor='black', linewidth=2)
ax.add_patch(z_circle)
ax.text(2, 8, 'z', fontsize=14, fontweight='bold', ha='center', va='center')

# Conditions c
c_circle = Circle((5.5, 8), 0.4, facecolor='white',
                  edgecolor='blue', linewidth=2)
ax.add_patch(c_circle)
ax.text(5.5, 8, 'c', fontsize=14, fontweight='bold', ha='center', va='center')

# Arrow from z to Generator
arrow_z_g = FancyArrowPatch((2.4, 8), (3.2, 7.2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
ax.add_patch(arrow_z_g)

# Arrow from c to Generator
arrow_c_g = FancyArrowPatch((5.1, 8), (4.8, 7.2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='blue')
ax.add_patch(arrow_c_g)

# Generator G
gen_box = FancyBboxPatch((3, 6.5), 3, 1.5,
                         boxstyle="round,pad=0.1",
                         facecolor=color_generator, alpha=0.7,
                         edgecolor='red', linewidth=2.5)
ax.add_patch(gen_box)
ax.text(4.5, 7.25, 'G', fontsize=18, fontweight='bold',
        ha='center', va='center', color='darkred')

# Arrow from Generator to Generated Video
arrow_g_video = FancyArrowPatch((4.5, 6.5), (4.5, 4),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2.5, color='black')
ax.add_patch(arrow_g_video)

# Generated Video G(z,c)
gen_video_circle = Circle((4.5, 3.5), 0.4, facecolor='white',
                         edgecolor='red', linewidth=2)
ax.add_patch(gen_video_circle)
ax.text(4.5, 3.5, 'G(z,c)', fontsize=11, fontweight='bold',
        ha='center', va='center')

# Real Video x
real_video_circle = Circle((2, 2.5), 0.4, facecolor='white',
                          edgecolor='black', linewidth=2)
ax.add_patch(real_video_circle)
ax.text(2, 2.5, 'x', fontsize=14, fontweight='bold',
        ha='center', va='center')

# Conditions c (for discriminator)
c2_circle = Circle((5.5, 2.5), 0.4, facecolor='white',
                   edgecolor='blue', linewidth=2)
ax.add_patch(c2_circle)
ax.text(5.5, 2.5, 'c', fontsize=14, fontweight='bold',
        ha='center', va='center')

# Arrow from Real Video x to Discriminator input
arrow_x_d = FancyArrowPatch((2.4, 2.5), (8.5, 2.2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
ax.add_patch(arrow_x_d)

# Arrow from Generated Video to Discriminator input
arrow_g_d = FancyArrowPatch((4.9, 3.5), (8.5, 2.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='red')
ax.add_patch(arrow_g_d)

# Arrow from Conditions c to Discriminator input
arrow_c_d = FancyArrowPatch((5.1, 2.5), (8.5, 2.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='blue')
ax.add_patch(arrow_c_d)

# Combined inputs to Discriminator
# Input 1: Generated + c
comb1_box = Rectangle((7.5, 2.6), 2, 0.3,
                      facecolor='lightyellow', alpha=0.7,
                      edgecolor='orange', linewidth=1.5)
ax.add_patch(comb1_box)
ax.text(8.5, 2.75, '[G(z,c), c]', fontsize=9, fontweight='bold',
        ha='center', va='center')

# Input 2: Real + c
comb2_box = Rectangle((7.5, 2.1), 2, 0.3,
                      facecolor='lightyellow', alpha=0.7,
                      edgecolor='orange', linewidth=1.5)
ax.add_patch(comb2_box)
ax.text(8.5, 2.25, '[x, c]', fontsize=9, fontweight='bold',
        ha='center', va='center')

# Discriminator D
disc_box = FancyBboxPatch((10.5, 1.5), 2.5, 2,
                          boxstyle="round,pad=0.1",
                          facecolor=color_discriminator, alpha=0.7,
                          edgecolor='orange', linewidth=2.5)
ax.add_patch(disc_box)
ax.text(11.75, 2.5, 'D', fontsize=18, fontweight='bold',
        ha='center', va='center', color='darkorange')

# Arrow from Discriminator to Output
arrow_d_out = FancyArrowPatch((13, 2.5), (14.5, 2.5),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='black')
ax.add_patch(arrow_d_out)

# Output P(y)
output_circle = Circle((15, 2.5), 0.4, facecolor='white',
                      edgecolor='orange', linewidth=2)
ax.add_patch(output_circle)
ax.text(15, 2.5, 'P(y)', fontsize=12, fontweight='bold',
        ha='center', va='center')

# Title
ax.text(8, 9.7, 'C3D-GAN Architecture', fontsize=20, fontweight='bold',
        ha='center', color='#2C3E50')

# Labels
ax.text(2, 8.6, 'Random Noise\n(z)', fontsize=10,
        ha='center', style='italic', color='black')
ax.text(5.5, 8.6, 'Conditions\n(c)', fontsize=10,
        ha='center', style='italic', color='blue')
ax.text(4.5, 6.2, 'Generator', fontsize=11, ha='center', style='italic', color='darkred')
ax.text(4.5, 2.8, 'Generated\nVideo', fontsize=10, ha='center', style='italic', color='red')
ax.text(2, 1.8, 'Real Video', fontsize=10, ha='center', style='italic', color='black')
ax.text(5.5, 1.8, 'Conditions\n(c)', fontsize=10, ha='center', style='italic', color='blue')
ax.text(11.75, 1.2, 'Discriminator', fontsize=11, ha='center', style='italic', color='darkorange')
ax.text(15, 2, 'Output\nScore', fontsize=10, ha='center', style='italic', color='orange')

# Add note about no encoder
ax.text(8, 0.2, 'Note: No Encoder needed! Only 2 networks (G + D) vs 3 in BiGAN',
        fontsize=11, ha='center', style='italic', color='darkgreen', weight='bold')

plt.tight_layout()
plt.savefig('c3dgan_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved C3D-GAN architecture diagram to: c3dgan_architecture_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created C3D-GAN architecture diagram!")
print("="*80)
print("\nShows:")
print("  - Features domain: Random noise z + Conditions c → Generator G")
print("  - Generator: Creates video G(z,c)")
print("  - Data domain: Generated video G(z,c) + Real video x + Conditions c")
print("  - Discriminator: Evaluates [G(z,c), c] and [x, c]")
print("  - Output: Real/Fake score P(y)")
print("\nKey difference: No Encoder! Only 2 networks vs 3 in BiGAN")
