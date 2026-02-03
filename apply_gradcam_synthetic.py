"""
GradCAM Visualization for Synthetic Echocardiogram Videos
Applies GradCAM to the discriminator to show which regions are important
for distinguishing synthetic videos.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Discriminator(nn.Module):
    """
    3D Discriminator with conditioning
    Modified to support GradCAM hooks
    """
    def __init__(self, cond_dim=11, size=64):
        super().__init__()
        self.size = size
        
        layers = []
        
        # Input layer
        if size == 128:
            layers.extend([
                nn.Conv3d(1, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            layers.extend([
                nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        elif size == 64:
            layers.extend([
                nn.Conv3d(1, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        else:
            layers.extend([
                nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # 32 -> 16
        layers.extend([
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 16 -> 8
        layers.extend([
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 8 -> 4 (This is our target layer for GradCAM)
        layers.extend([
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.features = nn.Sequential(*layers)
        
        # Classifier with conditioning
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4 + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, cond):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        x = torch.cat([features_flat, cond], dim=1)
        return self.classifier(x), features


class GradCAM:
    """
    GradCAM for 3D CNNs
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        # Use full backward hook for proper gradient capture
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_video, condition, class_idx=None):
        """
        Generate GradCAM heatmap for a video
        """
        self.model.eval()
        
        # Forward pass
        output, features = self.model(input_video, condition)
        
        if class_idx is None:
            # Use the output score (for discriminator, higher = more "real")
            score = output
        else:
            score = output[:, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, T, H, W]
        activations = self.activations  # [B, C, T, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, T, H, W]
        
        # ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()  # [T, H, W]
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def create_condition_from_filename(filename):
    """
    Extract condition from filename like: synth_0000_sexF_age0-1y_bmiunderweight.npy
    """
    parts = filename.replace('.npy', '').split('_')
    sex_str = None
    age_str = None
    bmi_str = None
    
    for part in parts:
        if part.startswith('sex'):
            sex_str = part[3:]
        elif part.startswith('age'):
            age_str = part[3:]
        elif part.startswith('bmi'):
            bmi_str = part[3:]
    
    # One-hot encoding
    sex_map = {'F': 0, 'M': 1}
    age_map = {'0-1y': 0, '2-5y': 1, '6-10y': 2, '11-15y': 3, '16-18y': 4}
    bmi_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
    
    sex_onehot = torch.zeros(2)
    if sex_str in sex_map:
        sex_onehot[sex_map[sex_str]] = 1
    
    age_onehot = torch.zeros(5)
    if age_str in age_map:
        age_onehot[age_map[age_str]] = 1
    
    bmi_onehot = torch.zeros(4)
    if bmi_str in bmi_map:
        bmi_onehot[bmi_map[bmi_str]] = 1
    
    cond = torch.cat([sex_onehot, age_onehot, bmi_onehot])
    return cond


def overlay_heatmap_on_video(video, heatmap, alpha=0.5):
    """
    Overlay heatmap on video frames
    video: [T, H, W] in range [-1, 1] or [0, 255]
    heatmap: [T, H, W] in range [0, 1]
    """
    # Normalize video to [0, 255]
    if video.min() < 0:
        video = (video + 1) * 127.5
    video = np.clip(video, 0, 255).astype(np.uint8)
    
    # Convert to RGB
    video_rgb = np.stack([video, video, video], axis=-1)  # [T, H, W, 3]
    
    # Upsample heatmap if needed (before coloring)
    T_video, H_video, W_video = video.shape
    T_heat, H_heat, W_heat = heatmap.shape
    
    if heatmap.shape != video.shape:
        # Upsample heatmap to match video dimensions
        heatmap_upsampled = np.zeros((T_video, H_video, W_video), dtype=np.float32)
        for t in range(min(T_video, T_heat)):
            if H_heat != H_video or W_heat != W_video:
                heatmap_upsampled[t] = cv2.resize(heatmap[t], (W_video, H_video), interpolation=cv2.INTER_LINEAR)
            else:
                heatmap_upsampled[t] = heatmap[t]
        # If video has more frames, repeat last heatmap frame
        if T_video > T_heat:
            for t in range(T_heat, T_video):
                heatmap_upsampled[t] = heatmap_upsampled[T_heat - 1]
        heatmap = heatmap_upsampled
    
    # Create colormap (jet: blue -> green -> yellow -> red)
    cmap = plt.cm.jet
    heatmap_colored = np.zeros((T_video, H_video, W_video, 3), dtype=np.uint8)
    for t in range(T_video):
        colored_frame = cmap(heatmap[t])[:, :, :3]  # Remove alpha channel
        heatmap_colored[t] = (colored_frame * 255).astype(np.uint8)
    
    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * video_rgb).astype(np.uint8)
    
    return overlay


def save_heatmap_frames(overlay, output_path, fps=30):
    """
    Save overlay frames as video
    """
    T, H, W, C = overlay.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for t in range(T):
        frame_bgr = cv2.cvtColor(overlay[t], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def visualize_sample_frames(video, heatmap, overlay, output_path, num_frames=8):
    """
    Create a grid visualization showing original, heatmap, and overlay
    """
    T_video = video.shape[0]
    T_overlay = overlay.shape[0]
    
    # Ensure we don't exceed available frames
    max_frames = min(T_video, T_overlay)
    num_frames = min(num_frames, max_frames)
    
    indices = np.linspace(0, max_frames-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 2, 6))
    
    # Normalize video for display
    if video.min() < 0:
        video_display = (video + 1) * 127.5
    else:
        video_display = video
    video_display = np.clip(video_display, 0, 255).astype(np.uint8)
    
    # Upsample heatmap if needed for visualization
    T_heat = heatmap.shape[0]
    H_heat, W_heat = heatmap.shape[1], heatmap.shape[2]
    H_video, W_video = video.shape[1], video.shape[2]
    
    if heatmap.shape != video.shape:
        heatmap_viz = np.zeros((T_video, H_video, W_video), dtype=np.float32)
        for t in range(min(T_video, T_heat)):
            if H_heat != H_video or W_heat != W_video:
                heatmap_viz[t] = cv2.resize(heatmap[t], (W_video, H_video), interpolation=cv2.INTER_LINEAR)
            else:
                heatmap_viz[t] = heatmap[t]
        if T_video > T_heat:
            for t in range(T_heat, T_video):
                heatmap_viz[t] = heatmap_viz[T_heat - 1]
        heatmap = heatmap_viz
    
    for i, idx in enumerate(indices):
        # Original frame
        axes[0, i].imshow(video_display[idx], cmap='gray')
        axes[0, i].set_title(f'Frame {idx}')
        axes[0, i].axis('off')
        
        # Heatmap
        axes[1, i].imshow(heatmap[idx], cmap='jet', vmin=0, vmax=1)
        axes[1, i].set_title(f'Heatmap {idx}')
        axes[1, i].axis('off')
        
        # Overlay
        axes[2, i].imshow(overlay[idx])
        axes[2, i].set_title(f'Overlay {idx}')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', rotation=90, labelpad=20)
    axes[1, 0].set_ylabel('Heatmap', rotation=90, labelpad=20)
    axes[2, 0].set_ylabel('Overlay', rotation=90, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Apply GradCAM to synthetic videos")
    parser.add_argument("--video_dir", type=str, default="final_videos",
                        help="Directory containing synthetic videos")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_c3dgan/discriminator_epoch0.pt",
                        help="Path to discriminator checkpoint")
    parser.add_argument("--output_dir", type=str, default="gradcam_results",
                        help="Output directory for heatmaps")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of videos to process")
    parser.add_argument("--size", type=int, default=64, choices=[32, 64, 128],
                        help="Video size")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Heatmap overlay transparency")
    
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Discriminator(cond_dim=11, size=args.size).to(device)
    
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")
    
    # Get target layer (last conv layer in features)
    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, nn.Conv3d):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find Conv3d layer in features")
    
    print(f"Target layer for GradCAM: {target_layer}")
    
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Get video files
    video_dir = Path(args.video_dir)
    video_files = sorted(list(video_dir.glob("*.npy")))[:args.num_videos]
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    # Process videos
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Load video
        video = np.load(video_path)  # [T, H, W]
        
        # Normalize to [-1, 1]
        video_tensor = torch.from_numpy(video).float()
        if video_tensor.max() > 1:
            video_tensor = video_tensor / 127.5 - 1.0
        
        # Add batch and channel dimensions: [1, 1, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Create condition from filename
        condition = create_condition_from_filename(video_path.name)
        condition = condition.unsqueeze(0).to(device)
        
        # Generate GradCAM
        heatmap = gradcam.generate_cam(video_tensor, condition)
        
        # Create overlay
        overlay = overlay_heatmap_on_video(video, heatmap, alpha=args.alpha)
        
        # Save overlay video
        video_name = video_path.stem
        overlay_path = output_dir / "overlays" / f"{video_name}_gradcam.mp4"
        save_heatmap_frames(overlay, str(overlay_path))
        
        # Save visualization
        viz_path = output_dir / "visualizations" / f"{video_name}_gradcam.png"
        visualize_sample_frames(video, heatmap, overlay, str(viz_path))
    
    print(f"\n✅ GradCAM complete!")
    print(f"Results saved in: {args.output_dir}/")
    print(f"  - Overlay videos: {args.output_dir}/overlays/")
    print(f"  - Visualizations: {args.output_dir}/visualizations/")


if __name__ == "__main__":
    main()

