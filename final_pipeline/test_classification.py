"""
Ultimate Validation: Classification Performance Test
Tests if synthetic data improves ML model accuracy

Compares:
1. Baseline: Model trained on REAL data only
2. Augmented: Model trained on REAL + SYNTHETIC data

If augmented model performs BETTER → Synthetic data is GOOD! ✅
If augmented model performs WORSE → Synthetic data is BAD! ❌
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VideoDataset(Dataset):
    """Dataset for classification"""
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video
        video = np.load(self.video_paths[idx])
        
        # Normalize
        video = video.astype(np.float32) / 127.5 - 1.0
        
        # To tensor [1, T, H, W]
        x = torch.from_numpy(video).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y


class VideoClassifier(nn.Module):
    """Simple 3D CNN classifier"""
    def __init__(self, num_classes=5):  # 5 age bins
        super().__init__()
        
        # 3D CNN feature extractor
        self.features = nn.Sequential(
            # 64 -> 32
            nn.Conv3d(1, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 8 -> 4
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def prepare_real_data(manifest_path):
    """Prepare real data for classification"""
    df = pd.read_csv(manifest_path)
    df = df[df['processed_path'].notna()].reset_index(drop=True)
    
    # Extract paths and age labels
    paths = df['processed_path'].tolist()
    
    # Map age bins to integers
    age_map = {'0-1': 0, '2-5': 1, '6-10': 2, '11-15': 3, '16-18': 4}
    labels = [age_map.get(str(row['age_bin']), 0) for _, row in df.iterrows()]
    
    return paths, labels


def prepare_synthetic_data(synthetic_dir):
    """Prepare synthetic data - extract labels from filenames"""
    paths = []
    labels = []
    
    age_map = {'0-1y': 0, '2-5y': 1, '6-10y': 2, '11-15y': 3, '16-18y': 4}
    
    for npy_file in Path(synthetic_dir).glob("*.npy"):
        # Parse filename: synth_0000_sexF_age0-1y_bmiNormal.npy
        filename = npy_file.stem
        
        # Extract age from filename
        for age_str, age_idx in age_map.items():
            if age_str in filename:
                paths.append(str(npy_file))
                labels.append(age_idx)
                break
    
    return paths, labels


def train_model(train_loader, val_loader, device, epochs=10):
    """Train classification model"""
    model = VideoClassifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                
                outputs = model(videos)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    return best_val_acc


def main():
    print("="*70)
    print("CLASSIFICATION EXPERIMENT: Real vs Real+Synthetic")
    print("="*70)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Prepare real data
    print("\n[1/5] Loading real data...")
    real_paths, real_labels = prepare_real_data("data_numpy/manifest.csv")
    print(f"  Loaded {len(real_paths)} real videos")
    
    # Split real data: 80% train, 20% validation (same for both experiments)
    X_train_real, X_val, y_train_real, y_val = train_test_split(
        real_paths, real_labels, test_size=0.2, random_state=42, stratify=real_labels
    )
    
    print(f"  Train: {len(X_train_real)}, Validation: {len(X_val)}")
    
    # Prepare synthetic data
    print("\n[2/5] Loading synthetic data...")
    synth_paths, synth_labels = prepare_synthetic_data("final_videos")
    print(f"  Loaded {len(synth_paths)} synthetic videos")
    
    # Combine real + synthetic for augmented training
    X_train_augmented = X_train_real + synth_paths
    y_train_augmented = y_train_real + synth_labels
    
    print(f"  Augmented train set: {len(X_train_augmented)} videos")
    print(f"    - Real: {len(X_train_real)}")
    print(f"    - Synthetic: {len(synth_paths)}")
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    train_real_dataset = VideoDataset(X_train_real, y_train_real)
    train_augmented_dataset = VideoDataset(X_train_augmented, y_train_augmented)
    val_dataset = VideoDataset(X_val, y_val)
    
    batch_size = 8
    train_real_loader = DataLoader(train_real_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_augmented_loader = DataLoader(train_augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Experiment 1: Real data only
    print("\n[4/5] Training BASELINE model (Real data only)...")
    print("="*70)
    baseline_acc = train_model(train_real_loader, val_loader, device, epochs=10)
    
    # Experiment 2: Real + Synthetic
    print("\n[5/5] Training AUGMENTED model (Real + Synthetic)...")
    print("="*70)
    augmented_acc = train_model(train_augmented_loader, val_loader, device, epochs=10)
    
    # Results
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    print(f"\nBaseline (Real only):     {baseline_acc:.2f}%")
    print(f"Augmented (Real+Synth):   {augmented_acc:.2f}%")
    print(f"Improvement:              {augmented_acc - baseline_acc:+.2f}%")
    
    if augmented_acc > baseline_acc + 1.0:
        print("\n✅ VERDICT: Synthetic data IMPROVES performance!")
        print("   → Synthetic videos are CORRECT and USEFUL for augmentation!")
        print(f"   → Performance gain: {augmented_acc - baseline_acc:.2f}%")
    elif augmented_acc > baseline_acc - 1.0:
        print("\n⚠️ VERDICT: Synthetic data has NEUTRAL effect")
        print("   → May be useful but not significantly helpful")
    else:
        print("\n❌ VERDICT: Synthetic data HURTS performance")
        print("   → Do NOT use for augmentation")
    
    print("="*70)


if __name__ == "__main__":
    main()






