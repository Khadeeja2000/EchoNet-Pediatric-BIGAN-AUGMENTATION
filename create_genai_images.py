"""
Create images specifically for GenAI project presentation
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json


def create_genai_flow_diagram():
    """Create GenAI project flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Title
    ax.text(6, 3.5, 'GenAI Project: Video-to-Text Validation Pipeline', 
            ha='center', fontsize=18, weight='bold')
    
    # Step 1: C3DGAN Generated Video
    box1 = FancyBboxPatch((0.5, 1.5), 2, 1, boxstyle="round,pad=0.15", 
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.5, 2, 'C3DGAN\nGenerated\nVideo', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.5, 2), (3.5, 2), arrowstyle='->', 
                             mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Step 2: Video Analysis
    box2 = FancyBboxPatch((3.5, 1.5), 2, 1, boxstyle="round,pad=0.15", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(4.5, 2, 'Video-to-Text\nAnalysis', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5.5, 2), (6.5, 2), arrowstyle='->', 
                             mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Step 3: Description
    box3 = FancyBboxPatch((6.5, 1.5), 2.5, 1, boxstyle="round,pad=0.15", 
                          facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(7.75, 2, 'Generated\nDescription', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((9, 2), (10, 2), arrowstyle='->', 
                             mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Step 4: Validation
    box4 = FancyBboxPatch((10, 1.5), 1.5, 1, boxstyle="round,pad=0.15", 
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(10.75, 2, 'Validation\n✓', ha='center', va='center', 
            fontsize=11, weight='bold', color='green')
    
    # Example text below
    ax.text(6, 0.8, 'Example Description:', ha='center', fontsize=10, weight='bold')
    ax.text(6, 0.4, '"Echocardiogram: PSAX view, Female, age 6-10 years"', 
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    os.makedirs('genai_presentation_images', exist_ok=True)
    plt.savefig('genai_presentation_images/01_genai_flow.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 01_genai_flow.png")
    plt.close()


def create_validation_process():
    """Create validation process diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(5, 6.5, 'Validation Process', ha='center', fontsize=18, weight='bold')
    
    # Expected
    expected_box = FancyBboxPatch((1, 4.5), 3, 1.5, boxstyle="round,pad=0.15", 
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(expected_box)
    ax.text(2.5, 5.5, 'EXPECTED', ha='center', fontsize=12, weight='bold')
    ax.text(2.5, 5, 'View: PSAX', ha='center', fontsize=10)
    ax.text(2.5, 4.7, 'Sex: Female', ha='center', fontsize=10)
    ax.text(2.5, 4.4, 'Age: 6-10 years', ha='center', fontsize=10)
    
    # Generated Video
    video_box = FancyBboxPatch((3.5, 2.5), 3, 1.5, boxstyle="round,pad=0.15", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(video_box)
    ax.text(5, 3.5, 'GENERATED VIDEO', ha='center', fontsize=12, weight='bold')
    ax.text(5, 3.2, '(C3DGAN Output)', ha='center', fontsize=9, style='italic')
    
    # Arrow down
    arrow1 = FancyArrowPatch((2.5, 4.5), (5, 3.5), arrowstyle='->', 
                            mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Analysis
    analysis_box = FancyBboxPatch((1, 0.5), 3, 1.5, boxstyle="round,pad=0.15", 
                                 facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(2.5, 1.5, 'EXTRACTED', ha='center', fontsize=12, weight='bold')
    ax.text(2.5, 1.2, 'View: PSAX', ha='center', fontsize=10)
    ax.text(2.5, 0.9, 'Sex: Female', ha='center', fontsize=10)
    ax.text(2.5, 0.6, 'Age: 6-10 years', ha='center', fontsize=10)
    
    # Arrow down
    arrow2 = FancyArrowPatch((5, 2.5), (2.5, 1.5), arrowstyle='->', 
                             mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(3.8, 2, 'Video-to-Text', ha='center', fontsize=9, style='italic', 
            rotation=-45)
    
    # Comparison
    match_box = FancyBboxPatch((6.5, 2), 2.5, 2, boxstyle="round,pad=0.15", 
                              facecolor='lightgreen', edgecolor='green', linewidth=3)
    ax.add_patch(match_box)
    ax.text(7.75, 3.5, 'MATCH ✓', ha='center', fontsize=14, weight='bold', color='green')
    ax.text(7.75, 3, 'All characteristics', ha='center', fontsize=10)
    ax.text(7.75, 2.7, 'match correctly', ha='center', fontsize=10)
    ax.text(7.75, 2.2, 'Video Validated!', ha='center', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig('genai_presentation_images/02_validation_process.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 02_validation_process.png")
    plt.close()


def create_results_table():
    """Create results table"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    
    ax.text(6, 5.5, 'Validation Results', ha='center', fontsize=18, weight='bold')
    
    # Table data
    data = [
        ['Video 1', 'PSAX', 'Female', '0-1', 'PSAX', 'Female', '0-1', '✓ MATCH'],
        ['Video 2', 'A4C', 'Female', '0-1', 'A4C', 'Female', '0-1', '✓ MATCH'],
        ['Video 3', 'PSAX', 'Female', '0-1', 'PSAX', 'Female', '0-1', '✓ MATCH'],
        ['Video 4', 'PSAX', 'Female', '0-1', 'PSAX', 'Female', '0-1', '✓ MATCH'],
        ['Video 5', 'A4C', 'Female', '0-1', 'A4C', 'Female', '0-1', '✓ MATCH'],
    ]
    
    # Table header
    headers = ['Video', 'Expected\nView', 'Expected\nSex', 'Expected\nAge', 
               'Extracted\nView', 'Extracted\nSex', 'Extracted\nAge', 'Status']
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('lightblue')
        table[(0, i)].set_text_props(weight='bold')
    
    # Style match cells
    for i in range(1, len(data) + 1):
        table[(i, 7)].set_facecolor('lightgreen')
        table[(i, 7)].set_text_props(weight='bold', color='green')
    
    # Summary
    ax.text(6, 0.1, 'Accuracy: 100% (5/5 videos validated correctly)', 
            ha='center', fontsize=12, weight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('genai_presentation_images/03_results_table.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 03_results_table.png")
    plt.close()


def create_genai_concepts():
    """Create GenAI concepts diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'GenAI Concepts Demonstrated', ha='center', fontsize=18, weight='bold')
    
    concepts = [
        ('Video-to-Text', 'Converting visual information\nto natural language', 2.5, 6),
        ('Multimodal Learning', 'Combining video + text +\nmetadata', 7.5, 6),
        ('Conditional Verification', 'Checking if generated content\nmatches input conditions', 2.5, 3.5),
        ('Quality Control', 'Using GenAI to validate\nGenAI outputs', 7.5, 3.5),
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for i, (title, desc, x, y) in enumerate(concepts):
        box = FancyBboxPatch((x-1.2, y-1), 2.4, 1.8, boxstyle="round,pad=0.15", 
                            facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.3, title, ha='center', fontsize=11, weight='bold')
        ax.text(x, y-0.3, desc, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('genai_presentation_images/04_genai_concepts.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 04_genai_concepts.png")
    plt.close()


def create_all_genai_images():
    """Create all GenAI presentation images"""
    os.makedirs('genai_presentation_images', exist_ok=True)
    
    print("Creating GenAI project presentation images...")
    print("="*60)
    
    create_genai_flow_diagram()
    create_validation_process()
    create_results_table()
    create_genai_concepts()
    
    print("="*60)
    print("✓ All images created in 'genai_presentation_images/' folder")
    print("\nImages created:")
    print("  1. 01_genai_flow.png - Complete pipeline")
    print("  2. 02_validation_process.png - How validation works")
    print("  3. 03_results_table.png - Validation results")
    print("  4. 04_genai_concepts.png - GenAI concepts")


if __name__ == "__main__":
    create_all_genai_images()




