"""
Quick GPU check - verify your Mac GPU is ready for training
Run this anytime: python3 check_gpu.py
"""
import torch
import time

print("="*60)
print("🔍 GPU STATUS CHECK")
print("="*60)

print(f"\nPyTorch: {torch.__version__}")

if torch.backends.mps.is_available():
    device = "mps"
    print(f"✅ Apple Silicon GPU (MPS): AVAILABLE")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"✅ NVIDIA GPU (CUDA): AVAILABLE")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"⚠️  GPU: NOT AVAILABLE (will use CPU)")

print(f"\n🚀 Training will use: {device.upper()}")

# Quick performance test
if device != "cpu":
    print("\n⏱️  Quick performance test...")
    x = torch.randn(2, 1, 32, 64, 64, device=device)
    start = time.time()
    for _ in range(5):
        y = torch.nn.functional.conv3d(x, torch.randn(16, 1, 3, 3, 3, device=device))
    elapsed = time.time() - start
    print(f"✓ 5 convolutions: {elapsed:.3f}s (GPU working!)")

print("\n" + "="*60)
print("✅ READY FOR TRAINING")
print("="*60)

if device == "mps":
    print("\n💡 Expected speedup: 3-5x faster than CPU")
    print("💡 Monitor GPU: Activity Monitor > Window > GPU History")

print("\n🚀 Start training:")
print("   python3 augmentation/train_bigan_improved.py --epochs 30")
print()

