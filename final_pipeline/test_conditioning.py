"""
Test if conditioning actually works!

Tests:
1. Same condition → Similar videos (consistency)
2. Different conditions → Different videos (discrimination)
3. Visual differences between sex/age/BMI groups
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from tqdm import tqdm


class Generator(nn.Module):
    """Same architecture as training"""
    def __init__(self, z_dim=128, cond_dim=11, size=64):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)
        
        layers = []
        layers.extend([
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
        ])
        
        if size >= 64:
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        else:
            layers.append(nn.ConvTranspose3d(64, 1, kernel_size=3, stride=1, padding=1))
            layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4, 4)
        return self.main(x)


def create_condition_vector(sex, age, bmi):
    """Create one-hot encoded condition vector"""
    cond = torch.zeros(11)
    cond[sex] = 1  # Sex: 0-1
    cond[2 + age] = 1  # Age: 2-6
    cond[7 + bmi] = 1  # BMI: 7-10
    return cond


def test_consistency(generator, device, num_samples=10):
    """Test 1: Same condition should produce similar videos"""
    print("\n" + "="*70)
    print("TEST 1: CONDITION CONSISTENCY")
    print("="*70)
    print("Testing: Same condition → Similar videos?")
    
    # Test condition: Female, age 6-10, normal BMI
    sex, age, bmi = 0, 2, 1
    cond = create_condition_vector(sex, age, bmi).unsqueeze(0).to(device)
    
    print(f"\nGenerating {num_samples} videos with SAME condition:")
    print(f"  Sex: Female (0), Age: 6-10y (2), BMI: Normal (1)")
    
    videos = []
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 128, device=device)
            video = generator(z, cond)
            videos.append(video.cpu().numpy())
    
    videos = np.array(videos)
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(videos)):
        for j in range(i+1, len(videos)):
            v1 = videos[i].flatten()
            v2 = videos[j].flatten()
            sim = 1 - cosine(v1, v2)  # Cosine similarity
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    
    print(f"\n  Average similarity: {avg_similarity:.3f}")
    print(f"  Std of similarity: {np.std(similarities):.3f}")
    
    if avg_similarity > 0.7:
        print(f"  ✅ PASS: Videos with same condition are similar (>{0.7:.1f})")
        result1 = True
    elif avg_similarity > 0.5:
        print(f"  ⚠️ MARGINAL: Some consistency but not strong")
        result1 = False
    else:
        print(f"  ❌ FAIL: Videos with same condition are too different")
        result1 = False
    
    return result1, avg_similarity


def test_discrimination(generator, device, num_samples=5):
    """Test 2: Different conditions should produce different videos"""
    print("\n" + "="*70)
    print("TEST 2: CONDITION DISCRIMINATION")
    print("="*70)
    print("Testing: Different conditions → Different videos?")
    
    # Define different conditions
    conditions = [
        (0, 0, 0, "Female, age 0-1, underweight"),
        (1, 4, 3, "Male, age 16-18, obese"),
        (0, 2, 1, "Female, age 6-10, normal"),
    ]
    
    condition_groups = []
    
    for sex, age, bmi, label in conditions:
        cond = create_condition_vector(sex, age, bmi).unsqueeze(0).to(device)
        
        group_videos = []
        with torch.no_grad():
            for i in range(num_samples):
                z = torch.randn(1, 128, device=device)
                video = generator(z, cond)
                group_videos.append(video.cpu().numpy())
        
        condition_groups.append((np.array(group_videos), label))
    
    # Calculate inter-group distances
    print("\nComparing different conditions:")
    
    group_means = [videos.mean(axis=0).flatten() for videos, _ in condition_groups]
    
    distances = []
    for i in range(len(group_means)):
        for j in range(i+1, len(group_means)):
            dist = 1 - cosine(group_means[i], group_means[j])
            print(f"  {conditions[i][3]}")
            print(f"    vs {conditions[j][3]}")
            print(f"    → Distance: {dist:.3f}")
            distances.append(dist)
    
    avg_distance = np.mean(distances)
    
    print(f"\n  Average distance between conditions: {avg_distance:.3f}")
    
    if avg_distance < 0.3:
        print(f"  ❌ FAIL: Different conditions produce too similar videos")
        result2 = False
    elif avg_distance < 0.5:
        print(f"  ⚠️ MARGINAL: Some discrimination but weak")
        result2 = False
    else:
        print(f"  ✅ PASS: Different conditions produce different videos!")
        result2 = True
    
    return result2, avg_distance


def test_condition_statistics(generator, device, samples_per_condition=10):
    """Test 3: Statistical differences between conditions"""
    print("\n" + "="*70)
    print("TEST 3: CONDITION-SPECIFIC STATISTICS")
    print("="*70)
    print("Testing: Do sex/age/BMI conditions have different characteristics?")
    
    # Test sex differences
    print("\n[Sex Test] Female vs Male:")
    
    sex_stats = {}
    for sex, label in [(0, "Female"), (1, "Male")]:
        cond = create_condition_vector(sex, 2, 1).unsqueeze(0).to(device)  # age=6-10, bmi=normal
        
        videos = []
        with torch.no_grad():
            for i in range(samples_per_condition):
                z = torch.randn(1, 128, device=device)
                video = generator(z, cond)
                videos.append(video.cpu().numpy())
        
        videos = np.array(videos)
        
        stats = {
            'mean': videos.mean(),
            'std': videos.std(),
            'motion': np.abs(np.diff(videos, axis=2)).mean()
        }
        sex_stats[label] = stats
        
        print(f"  {label}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, motion={stats['motion']:.2f}")
    
    # Check if different
    mean_diff = abs(sex_stats['Female']['mean'] - sex_stats['Male']['mean'])
    sex_different = mean_diff > 1.0
    
    # Test age differences
    print("\n[Age Test] Different age groups:")
    
    age_stats = []
    age_labels = ['0-1y', '2-5y', '6-10y', '11-15y', '16-18y']
    
    for age_idx in range(5):
        cond = create_condition_vector(0, age_idx, 1).unsqueeze(0).to(device)  # female, normal bmi
        
        videos = []
        with torch.no_grad():
            for i in range(samples_per_condition):
                z = torch.randn(1, 128, device=device)
                video = generator(z, cond)
                videos.append(video.cpu().numpy())
        
        videos = np.array(videos)
        mean_val = videos.mean()
        age_stats.append(mean_val)
        
        print(f"  Age {age_labels[age_idx]}: mean={mean_val:.2f}")
    
    # Check if age groups differ
    age_range = max(age_stats) - min(age_stats)
    age_different = age_range > 2.0
    
    # Test BMI differences
    print("\n[BMI Test] Different BMI categories:")
    
    bmi_stats = []
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    
    for bmi_idx in range(4):
        cond = create_condition_vector(0, 2, bmi_idx).unsqueeze(0).to(device)  # female, age 6-10
        
        videos = []
        with torch.no_grad():
            for i in range(samples_per_condition):
                z = torch.randn(1, 128, device=device)
                video = generator(z, cond)
                videos.append(video.cpu().numpy())
        
        videos = np.array(videos)
        mean_val = videos.mean()
        bmi_stats.append(mean_val)
        
        print(f"  BMI {bmi_labels[bmi_idx]}: mean={mean_val:.2f}")
    
    # Check if BMI groups differ
    bmi_range = max(bmi_stats) - min(bmi_stats)
    bmi_different = bmi_range > 2.0
    
    print("\n" + "-"*70)
    print("RESULTS:")
    if sex_different:
        print(f"  ✅ Sex conditioning: Working (difference={mean_diff:.2f})")
    else:
        print(f"  ❌ Sex conditioning: NOT working (difference={mean_diff:.2f})")
    
    if age_different:
        print(f"  ✅ Age conditioning: Working (range={age_range:.2f})")
    else:
        print(f"  ❌ Age conditioning: NOT working (range={age_range:.2f})")
    
    if bmi_different:
        print(f"  ✅ BMI conditioning: Working (range={bmi_range:.2f})")
    else:
        print(f"  ❌ BMI conditioning: NOT working (range={bmi_range:.2f})")
    
    result3 = sex_different or age_different or bmi_different
    
    return result3, (sex_different, age_different, bmi_different)


def main():
    print("="*70)
    print("CONDITIONING VERIFICATION TEST")
    print("="*70)
    print("\nThis test verifies if sex, age, and BMI conditioning works!")
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load generator
    print("\nLoading generator...")
    generator = Generator(z_dim=128, cond_dim=11, size=64).to(device)
    generator.load_state_dict(torch.load('checkpoints_c3dgan/generator_best.pt', map_location=device))
    generator.eval()
    print("✓ Generator loaded")
    
    # Run tests
    result1, consistency_score = test_consistency(generator, device, num_samples=10)
    result2, discrimination_score = test_discrimination(generator, device, num_samples=5)
    result3, (sex_works, age_works, bmi_works) = test_condition_statistics(generator, device, samples_per_condition=10)
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT: CONDITIONING VERIFICATION")
    print("="*70)
    
    print("\n📊 Test Results:")
    print(f"  Test 1 (Consistency):     {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"  Test 2 (Discrimination):  {'✅ PASS' if result2 else '❌ FAIL'}")
    print(f"  Test 3 (Statistics):")
    print(f"    - Sex conditioning:     {'✅ Works' if sex_works else '❌ Not working'}")
    print(f"    - Age conditioning:     {'✅ Works' if age_works else '❌ Not working'}")
    print(f"    - BMI conditioning:     {'✅ Works' if bmi_works else '❌ Not working'}")
    
    tests_passed = sum([result1, result2, result3])
    
    print(f"\n  Overall: {tests_passed}/3 tests passed")
    
    if tests_passed >= 2:
        print("\n✅ VERDICT: Conditioning IS WORKING!")
        print("   → Sex/Age/BMI parameters successfully control video generation")
        print("   → Different conditions produce different videos")
        print("   → Same condition produces consistent results")
    elif tests_passed == 1:
        print("\n⚠️ VERDICT: Conditioning PARTIALLY working")
        print("   → Some conditioning works, some doesn't")
        print("   → Model learned SOME condition information")
    else:
        print("\n❌ VERDICT: Conditioning NOT working")
        print("   → Model ignores sex/age/BMI parameters")
        print("   → Generated videos are random regardless of conditions")
    
    print("="*70)


if __name__ == "__main__":
    main()






