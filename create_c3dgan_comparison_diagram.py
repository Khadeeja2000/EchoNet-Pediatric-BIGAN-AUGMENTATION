"""
Create simple comparison diagram: C3D-GAN vs Other GANs
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('C3D-GAN vs Other GANs - Simple Comparison', fontsize=16, fontweight='bold', y=0.98)

# Colors
color_2d = '#FFE8E8'
color_3d = '#E8FFE8'
color_bigan = '#FFE8FF'

# ========== STANDARD GAN (2D) ==========
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.text(5, 2.7, 'STANDARD GAN (2D)', fontsize=12, fontweight='bold', ha='center')

# Noise
z_box = Rectangle((1, 1.5), 1, 0.5, facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax1.add_patch(z_box)
ax1.text(1.5, 1.75, 'z', fontsize=10, fontweight='bold', ha='center', va='center')

# Arrow
arrow1 = FancyArrowPatch((2.1, 1.75), (3.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax1.add_patch(arrow1)

# Generator
gen_box = FancyBboxPatch((3.5, 1.3), 1.5, 0.9, boxstyle="round,pad=0.1",
                         facecolor=color_2d, edgecolor='red', linewidth=2)
ax1.add_patch(gen_box)
ax1.text(4.25, 1.75, 'Generator\n(Conv2D)', fontsize=9, fontweight='bold', ha='center', va='center')

# Arrow
arrow2 = FancyArrowPatch((5.1, 1.75), (6.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax1.add_patch(arrow2)

# Image
img_box = Rectangle((6.5, 1.5), 1, 0.5, facecolor='white', edgecolor='black', linewidth=1.5)
ax1.add_patch(img_box)
ax1.text(7, 1.75, 'Image\n[H×W]', fontsize=9, ha='center', va='center')

# Arrow
arrow3 = FancyArrowPatch((7.6, 1.75), (8.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax1.add_patch(arrow3)

# Discriminator
disc_box = FancyBboxPatch((8.5, 1.3), 1.2, 0.9, boxstyle="round,pad=0.1",
                          facecolor=color_2d, edgecolor='red', linewidth=2)
ax1.add_patch(disc_box)
ax1.text(9.1, 1.75, 'Discrim\n(Conv2D)', fontsize=9, fontweight='bold', ha='center', va='center')

# Labels
ax1.text(1.5, 1.2, 'Noise', fontsize=8, ha='center', style='italic')
ax1.text(4.25, 1.1, '2D Only', fontsize=8, ha='center', style='italic', color='red')
ax1.text(7, 1.2, 'Image', fontsize=8, ha='center', style='italic')
ax1.text(9.1, 1.1, '2D Only', fontsize=8, ha='center', style='italic', color='red')
ax1.text(5, 0.5, '❌ Only Images | ❌ No Temporal', fontsize=9, ha='center', color='red')

# ========== C3D-GAN (3D) ==========
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 3)
ax2.axis('off')
ax2.text(5, 2.7, 'C3D-GAN (3D)', fontsize=12, fontweight='bold', ha='center', color='darkgreen')

# Noise and Conditions
z_box2 = Rectangle((0.5, 1.5), 0.8, 0.5, facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax2.add_patch(z_box2)
ax2.text(0.9, 1.75, 'z', fontsize=10, fontweight='bold', ha='center', va='center')

c_box = Rectangle((1.5, 1.5), 0.8, 0.5, facecolor='lightblue', edgecolor='blue', linewidth=1.5)
ax2.add_patch(c_box)
ax2.text(1.9, 1.75, 'c', fontsize=10, fontweight='bold', ha='center', va='center')

# Arrow
arrow4 = FancyArrowPatch((2.4, 1.75), (3.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax2.add_patch(arrow4)

# Generator
gen_box2 = FancyBboxPatch((3.5, 1.1), 1.5, 1.3, boxstyle="round,pad=0.1",
                          facecolor=color_3d, edgecolor='green', linewidth=2.5)
ax2.add_patch(gen_box2)
ax2.text(4.25, 1.75, 'Generator\n(Conv3D)', fontsize=9, fontweight='bold', ha='center', va='center')
ax2.text(4.25, 1.45, 'Kernel:[4,4,4]', fontsize=7, ha='center', style='italic')
ax2.text(4.25, 1.3, '512→256→128→64', fontsize=7, ha='center', style='italic')
ax2.text(4.25, 1.15, 'BatchNorm3D', fontsize=7, ha='center', style='italic')

# Arrow
arrow5 = FancyArrowPatch((5.1, 1.75), (6.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax2.add_patch(arrow5)

# Video
vid_box = Rectangle((6.5, 1.5), 1, 0.5, facecolor='white', edgecolor='green', linewidth=1.5)
ax2.add_patch(vid_box)
ax2.text(7, 1.75, 'Video\n[T×H×W]', fontsize=9, ha='center', va='center')

# Arrow
arrow6 = FancyArrowPatch((7.6, 1.75), (8.4, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax2.add_patch(arrow6)

# Discriminator
disc_box2 = FancyBboxPatch((8.5, 1.1), 1.2, 1.3, boxstyle="round,pad=0.1",
                           facecolor=color_3d, edgecolor='green', linewidth=2.5)
ax2.add_patch(disc_box2)
ax2.text(9.1, 1.75, 'Discrim\n(Conv3D)', fontsize=9, fontweight='bold', ha='center', va='center')
ax2.text(9.1, 1.45, 'Kernel:[4,4,4]', fontsize=7, ha='center', style='italic')
ax2.text(9.1, 1.3, '1→64→128→256→512', fontsize=7, ha='center', style='italic')
ax2.text(9.1, 1.15, 'BatchNorm3D', fontsize=7, ha='center', style='italic')

# Labels
ax2.text(0.9, 1.2, 'Noise', fontsize=8, ha='center', style='italic')
ax2.text(1.9, 1.2, 'Cond', fontsize=8, ha='center', style='italic')
ax2.text(4.25, 1.1, '3D', fontsize=8, ha='center', style='italic', color='darkgreen', weight='bold')
ax2.text(7, 1.2, 'Video', fontsize=8, ha='center', style='italic')
ax2.text(9.1, 1.1, '3D', fontsize=8, ha='center', style='italic', color='darkgreen', weight='bold')
ax2.text(5, 0.5, '✅ Videos | ✅ Temporal | ✅ Simple (2 networks)', fontsize=9, ha='center', color='darkgreen')

# ========== BiGAN (3D but Complex) ==========
ax3 = axes[2]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 3)
ax3.axis('off')
ax3.text(5, 2.7, 'BiGAN (3D but Complex)', fontsize=12, fontweight='bold', ha='center', color='purple')

# Noise
z_box3 = Rectangle((1, 1.5), 1, 0.5, facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax3.add_patch(z_box3)
ax3.text(1.5, 1.75, 'z', fontsize=10, fontweight='bold', ha='center', va='center')

# Arrow
arrow7 = FancyArrowPatch((2.1, 1.75), (2.9, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax3.add_patch(arrow7)

# Generator
gen_box3 = FancyBboxPatch((3, 1.2), 1.2, 1.1, boxstyle="round,pad=0.1",
                          facecolor=color_bigan, edgecolor='purple', linewidth=2)
ax3.add_patch(gen_box3)
ax3.text(3.6, 1.75, 'G\n(Conv3D)', fontsize=9, fontweight='bold', ha='center', va='center')
ax3.text(3.6, 1.5, 'Kernel:[4,4,4]', fontsize=7, ha='center', style='italic')
ax3.text(3.6, 1.35, '512→256→128', fontsize=7, ha='center', style='italic')

# Arrow
arrow8 = FancyArrowPatch((4.3, 1.75), (4.9, 1.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax3.add_patch(arrow8)

# Video
vid_box2 = Rectangle((5, 1.5), 0.8, 0.5, facecolor='white', edgecolor='purple', linewidth=1.5)
ax3.add_patch(vid_box2)
ax3.text(5.4, 1.75, 'Video', fontsize=9, ha='center', va='center')

# Arrow down
arrow9 = FancyArrowPatch((5.4, 1.5), (5.4, 1.1), arrowstyle='->', mutation_scale=15, linewidth=2)
ax3.add_patch(arrow9)

# Encoder
enc_box = FancyBboxPatch((4.5, 0.4), 1.8, 0.7, boxstyle="round,pad=0.1",
                        facecolor=color_bigan, edgecolor='purple', linewidth=2)
ax3.add_patch(enc_box)
ax3.text(5.4, 0.75, 'Encoder (Conv3D)', fontsize=9, fontweight='bold', ha='center', va='center')
ax3.text(5.4, 0.55, 'Kernel:[4,4,4]', fontsize=7, ha='center', style='italic')
ax3.text(5.4, 0.45, '1→64→128→256→512', fontsize=7, ha='center', style='italic')

# Arrow up
arrow10 = FancyArrowPatch((5.4, 1.0), (5.4, 1.4), arrowstyle='->', mutation_scale=15, linewidth=2)
ax3.add_patch(arrow10)

# Arrow to Discriminator
arrow11 = FancyArrowPatch((6.3, 0.75), (7.4, 0.75), arrowstyle='->', mutation_scale=15, linewidth=2)
ax3.add_patch(arrow11)

# Discriminator
disc_box3 = FancyBboxPatch((7.5, 0.4), 1.2, 0.7, boxstyle="round,pad=0.1",
                          facecolor=color_bigan, edgecolor='purple', linewidth=2)
ax3.add_patch(disc_box3)
ax3.text(8.1, 0.75, 'D\n(Conv3D)', fontsize=9, fontweight='bold', ha='center', va='center')
ax3.text(8.1, 0.55, 'Kernel:[4,4,4]', fontsize=7, ha='center', style='italic')
ax3.text(8.1, 0.45, '1→64→128→256', fontsize=7, ha='center', style='italic')

# Labels
ax3.text(1.5, 1.2, 'Noise', fontsize=8, ha='center', style='italic')
ax3.text(3.6, 1.1, '3D', fontsize=8, ha='center', style='italic', color='purple')
ax3.text(5.4, 1.2, 'Video', fontsize=8, ha='center', style='italic')
ax3.text(5.4, 0.3, '3D', fontsize=8, ha='center', style='italic', color='purple')
ax3.text(8.1, 0.3, '3D', fontsize=8, ha='center', style='italic', color='purple')
ax3.text(5, 0.1, '✅ 3D | ❌ 3 Networks (Complex!) | ❌ Unstable', fontsize=9, ha='center', color='purple')

plt.tight_layout()
plt.savefig('c3dgan_comparison_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved comparison diagram to: c3dgan_comparison_diagram.png")
plt.close()

print("\n" + "="*80)
print("Created C3D-GAN Comparison Diagram!")
print("="*80)
print("\nShows:")
print("  1. Standard GAN (2D): Only images, no temporal")
print("  2. C3D-GAN (3D): Videos, temporal, simple (2 networks)")
print("  3. BiGAN (3D): Videos, temporal, but complex (3 networks)")

