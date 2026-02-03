#!/bin/bash

# Quick Start Script for Full Dataset Processing & C3DGAN Training
# This script guides you through the complete pipeline

set -e  # Exit on error

echo "=================================================================="
echo "FULL DATASET PROCESSING & C3DGAN TRAINING - QUICK START"
echo "=================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check dataset
echo -e "${YELLOW}[Step 1/5]${NC} Checking dataset availability..."
if [ -d "Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi" ]; then
    echo -e "${GREEN}✅ Dataset found!${NC}"
    python3 -c "
import os, pandas as pd
path = 'Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi'
a4c = os.path.join(path, 'A4C/FileList.csv')
psax = os.path.join(path, 'PSAX/FileList.csv')
if os.path.exists(a4c) and os.path.exists(psax):
    a4c_df = pd.read_csv(a4c)
    psax_df = pd.read_csv(psax)
    print(f'   A4C videos: {len(a4c_df)}')
    print(f'   PSAX videos: {len(psax_df)}')
    print(f'   Total: {len(a4c_df) + len(psax_df)}')
"
else
    echo -e "${RED}❌ Dataset not found!${NC}"
    echo "   Please download the dataset first."
    echo "   Expected path: Dataset/azcopy_darwin_amd64_10.30.1/EchoNet-Pediatric/..."
    exit 1
fi

echo ""
read -p "Press Enter to continue to preprocessing..."

# Step 2: Preprocess dataset
echo ""
echo -e "${YELLOW}[Step 2/5]${NC} Preprocessing FULL dataset..."
echo "   This will process ALL videos (may take several hours)"
echo "   Output: data_numpy_full/"
read -p "   Start preprocessing? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 full_dataset_pipeline.py \
        --output_dir data_numpy_full \
        --size 64 \
        --frames 32
else
    echo "   Skipping preprocessing. You can run it later with:"
    echo "   python3 full_dataset_pipeline.py --output_dir data_numpy_full"
fi

# Step 3: Check preprocessing results
echo ""
echo -e "${YELLOW}[Step 3/5]${NC} Checking preprocessing results..."
if [ -f "data_numpy_full/manifest.csv" ]; then
    echo -e "${GREEN}✅ Manifest found!${NC}"
    python3 -c "
import pandas as pd
df = pd.read_csv('data_numpy_full/manifest.csv')
print(f'   Processed videos: {len(df)}')
print(f'   Manifest: data_numpy_full/manifest.csv')
"
else
    echo -e "${RED}❌ Manifest not found!${NC}"
    echo "   Please complete preprocessing first."
    exit 1
fi

echo ""
read -p "Press Enter to continue to training..."

# Step 4: Train C3DGAN
echo ""
echo -e "${YELLOW}[Step 4/5]${NC} Training C3DGAN..."
echo "   This will train for 50 epochs (may take days)"
echo "   Checkpoints: checkpoints_c3dgan_full/"
read -p "   Start training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Starting training (this will run in background)..."
    echo "   Monitor progress: tail -f training_full.log"
    
    python3 final_pipeline/train_c3dgan.py \
        --manifest data_numpy_full/manifest.csv \
        --epochs 50 \
        --batch_size 16 \
        --size 64 \
        --lr_g 0.0002 \
        --lr_d 0.0002 \
        --checkpoint_dir checkpoints_c3dgan_full \
        2>&1 | tee training_full.log &
    
    echo "   Training started in background (PID: $!)"
    echo "   Use 'jobs' to check status"
    echo "   Use 'tail -f training_full.log' to monitor"
else
    echo "   Skipping training. You can run it later with:"
    echo "   python3 final_pipeline/train_c3dgan.py --manifest data_numpy_full/manifest.csv --epochs 50 --batch_size 16 --size 64 --checkpoint_dir checkpoints_c3dgan_full"
fi

# Step 5: Generate videos
echo ""
echo -e "${YELLOW}[Step 5/5]${NC} Generate synthetic videos..."
echo "   Wait for training to complete first!"
echo "   Then run:"
echo ""
echo "   python3 final_pipeline/generate_videos.py \\"
echo "       --checkpoint checkpoints_c3dgan_full/generator_best.pt \\"
echo "       --num_samples 500 \\"
echo "       --output_dir synthetic_videos_full \\"
echo "       --size 64"
echo ""

echo "=================================================================="
echo "✅ SETUP COMPLETE!"
echo "=================================================================="
echo ""
echo "Next steps:"
echo "1. Wait for preprocessing to complete (if running)"
echo "2. Wait for training to complete (if running)"
echo "3. Generate videos using the command above"
echo ""
echo "Monitor progress:"
echo "  - Preprocessing: Check data_numpy_full/manifest.csv"
echo "  - Training: tail -f training_full.log"
echo "  - Checkpoints: ls checkpoints_c3dgan_full/"
echo ""

