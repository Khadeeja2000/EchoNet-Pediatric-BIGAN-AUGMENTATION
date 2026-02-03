# Step-by-Step Guide: Full Dataset Processing & High-Quality C3DGAN Training

## Overview
This guide will help you process the **ENTIRE** EchoNet-Pediatric dataset and train C3DGAN to generate high-quality synthetic echocardiogram videos.

---

## Prerequisites

1. **Dataset Location**: Ensure the dataset is downloaded and accessible
   - Expected path: `Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/`
   - Should contain: `A4C/` and `PSAX/` folders with `FileList.csv` and `Videos/` subdirectories

2. **Python Environment**: 
   ```bash
   pip install torch torchvision numpy pandas opencv-python tqdm
   ```

3. **Disk Space**: 
   - Full dataset preprocessing requires significant disk space (~50-100GB depending on dataset size)
   - Processed numpy arrays: ~500MB-2GB per 1000 videos (at 64x64x32)

---

## Step 1: Check Dataset Availability

First, verify your dataset is accessible:

```bash
python3 -c "
import os
path = 'Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi'
if os.path.exists(path):
    print('✅ Dataset found!')
    a4c = os.path.join(path, 'A4C/FileList.csv')
    psax = os.path.join(path, 'PSAX/FileList.csv')
    if os.path.exists(a4c) and os.path.exists(psax):
        import pandas as pd
        a4c_df = pd.read_csv(a4c)
        psax_df = pd.read_csv(psax)
        print(f'   A4C videos: {len(a4c_df)}')
        print(f'   PSAX videos: {len(psax_df)}')
        print(f'   Total: {len(a4c_df) + len(psax_df)}')
    else:
        print('❌ FileList.csv files not found')
else:
    print('❌ Dataset path not found')
    print('   Please download the dataset first')
"
```

---

## Step 2: Preprocess FULL Dataset

Process **ALL** videos in the dataset (not just 2000):

```bash
# Basic preprocessing (64x64x32)
python3 full_dataset_pipeline.py

# With custom settings
python3 full_dataset_pipeline.py \
    --output_dir data_numpy_full \
    --size 64 \
    --frames 32

# Resume if interrupted (skips already processed videos)
python3 full_dataset_pipeline.py --resume

# Higher resolution for better quality (slower, more disk space)
python3 full_dataset_pipeline.py --size 128 --frames 32
```

**What this does:**
- Loads ALL videos from A4C and PSAX folders
- Filters videos with complete metadata (Sex, Age, Weight, Height)
- Preprocesses each video: resize to target size, extract frames, save as numpy
- Creates manifest CSV with all processed videos
- Supports resuming if interrupted

**Expected time:** 
- ~1-5 seconds per video depending on hardware
- For 10,000 videos: ~3-14 hours

**Output:**
- `data_numpy_full/manifest.csv` - Complete manifest of processed videos
- `data_numpy_full/*.npy` - Processed video arrays (one per video)

---

## Step 3: Train C3DGAN on Full Dataset

Train C3DGAN with optimal settings for high-quality generation:

```bash
# Standard training (64x64x32)
python3 final_pipeline/train_c3dgan.py \
    --manifest data_numpy_full/manifest.csv \
    --epochs 50 \
    --batch_size 16 \
    --size 64 \
    --lr_g 0.0002 \
    --lr_d 0.0002 \
    --checkpoint_dir checkpoints_c3dgan_full

# Higher resolution training (128x128x32) - Better quality but slower
python3 final_pipeline/train_c3dgan.py \
    --manifest data_numpy_full/manifest.csv \
    --epochs 50 \
    --batch_size 8 \
    --size 128 \
    --lr_g 0.0001 \
    --lr_d 0.0001 \
    --checkpoint_dir checkpoints_c3dgan_full_128

# Monitor training (in another terminal)
tail -f training.log
```

**Training Configuration:**
- **Epochs**: 50+ for full dataset (more data = more epochs needed)
- **Batch Size**: 
  - 64x64: 16-32 (faster)
  - 128x128: 8-16 (more memory)
- **Learning Rate**: 
  - 64x64: 0.0002 (standard)
  - 128x128: 0.0001 (more stable)
- **Resolution**: 
  - 64x64: Faster training, good quality
  - 128x128: Slower, better quality

**Expected time:**
- 64x64: ~2-5 hours per epoch (50 epochs = 4-10 days)
- 128x128: ~4-10 hours per epoch (50 epochs = 8-20 days)

**Checkpoints saved:**
- `checkpoints_c3dgan_full/generator_epoch{N}.pt`
- `checkpoints_c3dgan_full/discriminator_epoch{N}.pt`
- `checkpoints_c3dgan_full/generator_best.pt` (best model)

---

## Step 4: Generate High-Quality Synthetic Videos

Generate synthetic videos using the trained model:

```bash
# Generate 200 videos with all condition combinations
python3 final_pipeline/generate_videos.py \
    --checkpoint checkpoints_c3dgan_full/generator_best.pt \
    --num_samples 200 \
    --output_dir synthetic_videos_full \
    --size 64

# Generate more videos (500)
python3 final_pipeline/generate_videos.py \
    --checkpoint checkpoints_c3dgan_full/generator_best.pt \
    --num_samples 500 \
    --output_dir synthetic_videos_full \
    --size 64

# Generate from 128x128 model (higher quality)
python3 final_pipeline/generate_videos.py \
    --checkpoint checkpoints_c3dgan_full_128/generator_best.pt \
    --num_samples 200 \
    --output_dir synthetic_videos_full_128 \
    --size 128
```

**What this generates:**
- Videos with all combinations of:
  - Sex: F, M
  - Age: 0-1y, 2-5y, 6-10y, 11-15y, 16-18y
  - BMI: underweight, normal, overweight, obese
- Total: 2 × 5 × 4 = 40 unique combinations
- Multiple samples per combination for diversity

**Output:**
- `synthetic_videos_full/synth_XXXX_sexX_ageXy_bmiX.mp4` - Video files
- `synthetic_videos_full/synth_XXXX_sexX_ageXy_bmiX.npy` - Numpy arrays

---

## Step 5: Validate Generated Videos (Optional)

Apply GradCAM to validate synthetic video quality:

```bash
python3 apply_gradcam_synthetic.py \
    --video_dir synthetic_videos_full \
    --num_videos 100 \
    --checkpoint checkpoints_c3dgan_full/discriminator_epoch49.pt
```

---

## Complete Workflow Example

```bash
# 1. Check dataset
python3 -c "import os; print('✅' if os.path.exists('Dataset') else '❌')"

# 2. Preprocess (run overnight or in background)
nohup python3 full_dataset_pipeline.py --output_dir data_numpy_full > preprocessing.log 2>&1 &

# 3. Monitor preprocessing
tail -f preprocessing.log

# 4. Train (run for several days, use screen/tmux)
screen -S training
python3 final_pipeline/train_c3dgan.py \
    --manifest data_numpy_full/manifest.csv \
    --epochs 50 \
    --batch_size 16 \
    --size 64 \
    --checkpoint_dir checkpoints_c3dgan_full \
    2>&1 | tee training.log

# 5. Generate (after training completes)
python3 final_pipeline/generate_videos.py \
    --checkpoint checkpoints_c3dgan_full/generator_best.pt \
    --num_samples 500 \
    --output_dir synthetic_videos_full \
    --size 64
```

---

## Tips for Best Results

1. **Dataset Size**: More data = better quality, but longer training
   - Minimum: 5,000 videos
   - Recommended: 10,000+ videos
   - Optimal: All available videos

2. **Training Duration**: 
   - Monitor loss curves (D_loss and G_loss should stabilize)
   - Save checkpoints regularly
   - Use best model (lowest G_loss) for generation

3. **Resolution Trade-offs**:
   - 64x64: Faster, good for testing
   - 128x128: Better quality, requires more GPU memory

4. **Batch Size**:
   - Larger batch = more stable training
   - Adjust based on GPU memory

5. **Resuming**:
   - Preprocessing: Use `--resume` flag
   - Training: Modify script to load checkpoint and continue

---

## Troubleshooting

**Problem**: Dataset not found
- **Solution**: Check path, download dataset if missing

**Problem**: Out of memory during training
- **Solution**: Reduce batch_size or use smaller resolution

**Problem**: Training too slow
- **Solution**: Use 64x64 instead of 128x128, increase batch_size if possible

**Problem**: Low quality generated videos
- **Solution**: Train for more epochs, use higher resolution, ensure sufficient dataset size

**Problem**: Preprocessing interrupted
- **Solution**: Use `--resume` flag to continue from where it stopped

---

## Expected Results

After completing all steps, you should have:

1. ✅ Full dataset processed (~10,000+ videos)
2. ✅ Trained C3DGAN model (50+ epochs)
3. ✅ High-quality synthetic videos (200-500+ videos)
4. ✅ All condition combinations covered
5. ✅ GradCAM validation (optional)

The synthetic videos should show:
- Realistic cardiac motion
- Proper anatomical structures
- Smooth temporal transitions
- Good quality (comparable to real videos)

---

## Next Steps

1. Evaluate generated videos qualitatively
2. Apply GradCAM for validation
3. Use synthetic videos for data augmentation
4. Fine-tune model if needed
5. Generate more videos as needed

Good luck! 🚀

