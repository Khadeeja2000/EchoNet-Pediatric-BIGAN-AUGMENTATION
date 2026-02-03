"""
Example: How video content analysis is useful in practice
"""
import json

print("="*70)
print("PRACTICAL EXAMPLE: How Video Content Analysis is Useful")
print("="*70)

print("\nSCENARIO: C3DGAN generated 100 videos. How do we validate them?")
print("\n" + "-"*70)

# Example videos with different issues
examples = [
    {
        "video": "Video 1",
        "motion": 0.52,
        "edges": 0.132,
        "brightness": 21.2,
        "issue": None,
        "action": "✓ ACCEPT - All metrics normal"
    },
    {
        "video": "Video 2",
        "motion": 0.01,
        "edges": 0.05,
        "brightness": 5.0,
        "issue": "FROZEN/BLURRY/DARK",
        "action": "✗ REJECT - Video is frozen, blurry, and too dark"
    },
    {
        "video": "Video 3",
        "motion": 5.0,
        "edges": 0.5,
        "brightness": 200,
        "issue": "ABNORMAL",
        "action": "✗ REJECT - Motion too high, edges too dense, too bright"
    },
    {
        "video": "Video 4",
        "motion": 0.8,
        "edges": 0.15,
        "brightness": 80,
        "issue": None,
        "action": "✓ ACCEPT - Good quality video"
    }
]

print("\nAUTOMATED VALIDATION RESULTS:\n")
for ex in examples:
    print(f"{ex['video']}:")
    print(f"  Motion: {ex['motion']:.2f}, Edges: {ex['edges']:.3f}, Brightness: {ex['brightness']:.1f}")
    if ex['issue']:
        print(f"  ⚠ Issue detected: {ex['issue']}")
    print(f"  {ex['action']}")
    print()

print("-"*70)
print("\nWITHOUT CONTENT ANALYSIS:")
print("  ✗ Can only check metadata (might be wrong)")
print("  ✗ Can't detect quality issues")
print("  ✗ Can't tell if video is blurry/frozen/dark")
print("  ✗ Manual checking required (time-consuming)")
print("  ✗ Might use bad videos")

print("\nWITH CONTENT ANALYSIS:")
print("  ✓ Validates actual video content")
print("  ✓ Detects quality issues automatically")
print("  ✓ Filters out bad videos")
print("  ✓ Fully automated (saves time)")
print("  ✓ Only keeps high-quality videos")

print("\n" + "="*70)
print("REAL-WORLD BENEFITS:")
print("="*70)
print("\n1. TIME SAVING:")
print("   - Manual checking: 100 videos × 30 seconds = 50 minutes")
print("   - Automated analysis: 100 videos × 2 seconds = 3 minutes")
print("   - Saves 47 minutes!")

print("\n2. QUALITY IMPROVEMENT:")
print("   - Filters out 10-20% bad videos automatically")
print("   - Only keeps videos that meet quality standards")
print("   - Improves dataset quality")

print("\n3. RELIABILITY:")
print("   - Catches issues metadata check might miss")
print("   - Validates actual content, not just labels")
print("   - More thorough validation")

print("\n4. SCALABILITY:")
print("   - Can analyze thousands of videos")
print("   - No manual work needed")
print("   - Consistent quality control")

print("\n" + "="*70)
print("THIS IS WHY IT'S USEFUL!")
print("="*70)




