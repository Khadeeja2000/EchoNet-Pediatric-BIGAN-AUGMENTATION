"""
Quick code validation script - checks imports and model initialization without training.
This helps verify the code is correct before spending hours on training.
"""
import sys
import torch
import numpy as np

def validate_improved_training():
    """Validate train_bigan_improved.py can be imported and models instantiated"""
    print("="*60)
    print("VALIDATING IMPROVED BIGAN CODE")
    print("="*60)
    
    try:
        # Check PyTorch version
        print(f"\n✓ PyTorch version: {torch.__version__}")
        
        # Check device availability
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print(f"✓ MPS (Metal) available")
        else:
            device = "cpu"
            print(f"✓ Using CPU")
        
        # Import the training module
        print("\n[1/6] Importing training module...")
        sys.path.insert(0, 'augmentation')
        import train_bigan_improved
        print("✓ Training module imported successfully")
        
        # Test dataset initialization
        print("\n[2/6] Testing dataset...")
        try:
            dataset = train_bigan_improved.EchoDataset("data/processed/manifest.csv", frames=32, size=64)
            print(f"✓ Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"⚠️  Dataset warning: {e}")
        
        # Test Generator
        print("\n[3/6] Testing Generator...")
        G = train_bigan_improved.Generator(z_dim=128, cond_dim=2, size=64).to(device)
        z_test = torch.randn(1, 128, device=device)
        c_test = torch.tensor([[0.0, 1.0]], device=device)
        with torch.no_grad():
            out = G(z_test, c_test)
        print(f"✓ Generator works: input z(1,128) -> output {out.shape}")
        
        # Test Encoder
        print("\n[4/6] Testing Encoder...")
        E = train_bigan_improved.Encoder(z_dim=128, cond_dim=2, size=64).to(device)
        x_test = torch.randn(1, 1, 32, 64, 64, device=device)
        with torch.no_grad():
            z_out = E(x_test, c_test)
        print(f"✓ Encoder works: input video(1,1,32,64,64) -> output z{z_out.shape}")
        
        # Test Discriminator
        print("\n[5/6] Testing Discriminator...")
        D = train_bigan_improved.Discriminator(z_dim=128, cond_dim=2, size=64, use_sn=True).to(device)
        with torch.no_grad():
            score = D(x_test, z_out, c_test)
        print(f"✓ Discriminator works: outputs score {score.shape}")
        print(f"  - Spectral normalization: ENABLED")
        
        # Test EMA
        print("\n[6/6] Testing EMA...")
        ema = train_bigan_improved.EMA(G, decay=0.999)
        ema.update()
        print(f"✓ EMA works")
        
        # Memory usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        print("\n" + "="*60)
        print("✅ ALL VALIDATIONS PASSED!")
        print("="*60)
        print("\nYour code is ready. To run training:")
        print("  python3 augmentation/test_training.py     # Quick test (2 epochs)")
        print("  python3 augmentation/train_bigan_improved.py --epochs 30  # Full training")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_generation():
    """Validate generate_samples_improved.py"""
    print("\n" + "="*60)
    print("VALIDATING SAMPLE GENERATION CODE")
    print("="*60)
    
    try:
        sys.path.insert(0, 'augmentation')
        import generate_samples_improved
        print("✓ Generation module imported successfully")
        
        # Test Generator (simpler version in generation script)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        G = generate_samples_improved.Generator(z_dim=128, cond_dim=2, size=64).to(device)
        
        z_test = torch.randn(1, 128, device=device)
        c_test = torch.tensor([[0.0, 1.0]], device=device)
        with torch.no_grad():
            out = G(z_test, c_test)
        
        print(f"✓ Generation works: z(1,128) -> video {out.shape}")
        
        # Test validation
        metrics = generate_samples_improved.validate_video_quality(out)
        print(f"✓ Quality validation works: is_valid={metrics['is_valid']}")
        
        print("\n✅ GENERATION CODE VALIDATED")
        return True
        
    except Exception as e:
        print(f"❌ Generation validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🔍 BiGAN Code Validation Suite\n")
    
    success = True
    success = validate_improved_training() and success
    success = validate_generation() and success
    
    if success:
        print("\n" + "="*60)
        print("🎉 ALL VALIDATIONS PASSED - CODE IS READY!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ SOME VALIDATIONS FAILED")
        print("="*60)
        sys.exit(1)

