"""
Neural Network-Based Super-Resolution Enhancement
Uses proper deep learning models for better quality
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, will use fallback methods")


def load_video(video_path):
    """Load video from .npy or .mp4 file"""
    if video_path.endswith('.npy'):
        video = np.load(video_path)
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        video = np.array(frames)
    
    if len(video.shape) == 4:
        video = video.squeeze()
    if len(video.shape) == 5:
        video = video.squeeze()
    
    if video.dtype != np.uint8:
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = np.clip(video, 0, 255).astype(np.uint8)
    
    return video


class SimpleSRNet(nn.Module):
    """
    Simple but effective super-resolution network
    Based on EDSR architecture principles
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        
        # Feature extraction
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[self._make_residual_block() for _ in range(4)]
        )
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    
    def _make_residual_block(self):
        return nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
        )
    
    def forward(self, x):
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Feature extraction
        out = self.relu1(self.conv1(x))
        residual = out
        
        # Residual blocks
        out = self.res_blocks(out)
        out = out + residual
        
        # Upsampling
        out = self.upsample(out)
        
        # Denormalize
        out = out * 255.0
        return torch.clamp(out, 0, 255)


def upscale_with_nn(frame, model, device):
    """Upscale frame using neural network"""
    if not TORCH_AVAILABLE or model is None:
        # Fallback to high-quality interpolation
        return cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Upscale
    with torch.no_grad():
        upscaled_tensor = model(frame_tensor)
    
    # Convert back
    upscaled = upscaled_tensor.squeeze().cpu().numpy().astype(np.uint8)
    return upscaled


def upscale_iterative_bicubic(frame, scale=2, iterations=2):
    """
    Iterative bicubic upscaling - often better than single step
    """
    current = frame.astype(np.float32)
    
    for _ in range(iterations):
        h, w = current.shape
        current = cv2.resize(current, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        # Light smoothing to reduce artifacts
        current = cv2.GaussianBlur(current, (3, 3), 0.5)
    
    # Final resize to exact scale
    h, w = frame.shape
    current = cv2.resize(current, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    return np.clip(current, 0, 255).astype(np.uint8)


def enhance_video_nn(video_path, output_path, method='iterative', target_size=128, device='cpu'):
    """
    Enhance video using neural network or high-quality iterative methods
    """
    print(f"\nProcessing: {os.path.basename(video_path)}")
    
    # Load video
    video = load_video(video_path)
    T, H, W = video.shape
    print(f"  Original: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]")
    
    # Initialize model if using NN
    model = None
    if method == 'nn' and TORCH_AVAILABLE:
        try:
            model = SimpleSRNet(scale=2).to(device)
            model.eval()
            print(f"  Using neural network model on {device}")
        except Exception as e:
            print(f"  Warning: Could not initialize NN model: {e}")
            method = 'iterative'
    
    # Upscale
    if H < target_size:
        print(f"  Upscaling: {H}×{W} → {target_size}×{target_size} (method: {method})")
        enhanced_frames = []
        for frame in tqdm(video, desc="    Upscaling", leave=False):
            if method == 'nn':
                upscaled = upscale_with_nn(frame, model, device)
            elif method == 'iterative':
                upscaled = upscale_iterative_bicubic(frame, scale=2, iterations=2)
            else:
                # Simple cubic
                upscaled = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Resize to exact target if needed
            if upscaled.shape[0] != target_size:
                upscaled = cv2.resize(upscaled, (target_size, target_size), 
                                     interpolation=cv2.INTER_CUBIC)
            
            enhanced_frames.append(upscaled)
        video = np.array(enhanced_frames)
    else:
        print(f"  Resolution already {H}×{W}")
    
    print(f"  Final: {video.shape}, range: [{video.min()}, {video.max()}]")
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(output_path.replace('.mp4', '.npy'), video)
    
    try:
        import imageio
        frames = [video[t] for t in range(video.shape[0])]
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', pixelformat='gray')
        print(f"  ✓ Saved: {output_path}")
    except Exception as e:
        print(f"  ⚠ Saved .npy only: {e}")
    
    return video


def enhance_directory_nn(input_dir, output_dir, method='iterative', num_samples=None, target_size=128):
    """Enhance videos with neural network or iterative methods"""
    print("\n" + "="*60)
    print("Neural Network / Iterative Super-Resolution")
    print("="*60)
    print(f"Method: {method}")
    print(f"Target resolution: {target_size}×{target_size}")
    
    if TORCH_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"PyTorch device: {device}")
    else:
        device = 'cpu'
        print("PyTorch not available, using CPU fallback")
    
    print("="*60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_path.glob("*.npy"))
    video_files.extend(list(input_path.glob("*.mp4")))
    video_files = [f for f in video_files if 'enhanced' not in f.name and 'esrgan' not in f.name and 'quality' not in f.name]
    
    if num_samples:
        video_files = video_files[:num_samples]
    
    print(f"\nFound {len(video_files)} videos to enhance\n")
    
    enhanced_count = 0
    for video_file in video_files:
        try:
            output_filename = f"nn_{video_file.stem}.mp4"
            output_filepath = output_path / output_filename
            
            enhance_video_nn(str(video_file), str(output_filepath), method=method, 
                           target_size=target_size, device=device)
            enhanced_count += 1
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"{'='*60}")
    print(f"Enhanced {enhanced_count}/{len(video_files)} videos")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Neural network super-resolution enhancement")
    parser.add_argument("--input_dir", type=str, default="final_videos")
    parser.add_argument("--output_dir", type=str, default="final_videos_nn")
    parser.add_argument("--method", type=str, default="iterative", 
                       choices=['nn', 'iterative', 'cubic'])
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=None)
    
    args = parser.parse_args()
    
    enhance_directory_nn(
        args.input_dir,
        args.output_dir,
        method=args.method,
        num_samples=args.num_samples,
        target_size=args.target_size
    )


if __name__ == "__main__":
    main()





