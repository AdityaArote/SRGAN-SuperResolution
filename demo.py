"""
SRGAN Inference Demo Script
Upscale any image by 4x using trained SRGAN generator.
"""
import os
import argparse
import cv2
import numpy as np

if 'TL_BACKEND' not in os.environ:
    os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from srgan import SRGAN_g

def run_sr(image_path, weights_path, output_dir):
    tlx.files.exists_or_mkdir(output_dir)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"[*] Initializing SRGAN Generator (Backend: {tlx.BACKEND})...")
    G = SRGAN_g()
    G.init_build(tlx.nn.Input(shape=(1, 3, 96, 96)))

    print(f"[*] Loading weights from: {weights_path}")
    if weights_path.endswith('g_torch.npz') or tlx.BACKEND == 'torch':
        G.load_weights(weights_path)
    else:
        G.load_weights(weights_path, format='npz_dict')
    G.set_eval()

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image at {image_path}")

    h, w, c = img.shape
    print(f"[*] Input image size: {w}x{h} (channels: {c})")

    # Downscale to create LR input if larger than 256
    # Or super-resolve directly
    # Convert BGR to RGB for neural network
    lr_img = img
    lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    lr_tensor = (lr_rgb / 127.5) - 1.0
    lr_tensor = np.asarray(lr_tensor, dtype=np.float32)
    lr_tensor = np.transpose(lr_tensor, axes=[2, 0, 1])
    lr_tensor = lr_tensor[np.newaxis, :, :, :]
    lr_tensor = tlx.ops.convert_to_tensor(lr_tensor)

    print("[*] Running SRGAN super-resolution (4x)...")
    sr_tensor = G(lr_tensor)
    sr_np = tlx.ops.convert_to_numpy(sr_tensor)
    sr_np = np.transpose(sr_np[0], axes=[1, 2, 0])
    sr_rgb = np.clip((sr_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    sr_img = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

    # Bicubic comparison
    bicubic_img = cv2.resize(lr_img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    lr_out_path = os.path.join(output_dir, f"{base_name}_lr.png")
    bicubic_out_path = os.path.join(output_dir, f"{base_name}_bicubic.png")
    sr_out_path = os.path.join(output_dir, f"{base_name}_srgan.png")

    cv2.imwrite(lr_out_path, lr_img)
    cv2.imwrite(bicubic_out_path, bicubic_img)
    cv2.imwrite(sr_out_path, sr_img)

    print(f"[+] Output generated:")
    print(f"    - Input LR:        {lr_out_path} ({w}x{h})")
    print(f"    - Bicubic 4x:      {bicubic_out_path} ({w * 4}x{h * 4})")
    print(f"    - SRGAN Result:    {sr_out_path} ({sr_img.shape[1]}x{sr_img.shape[0]})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SRGAN super-resolution on an image")
    parser.add_argument('--image', type=str, default='img/model.jpeg', help='Path to input image')
    parser.add_argument('--weights', type=str, default='models/g_torch.npz', help='Path to weights file')
    parser.add_argument('--output_dir', type=str, default='samples', help='Directory to save outputs')

    args = parser.parse_args()

    # Fallback to g.npz if g_torch.npz doesn't exist
    if not os.path.exists(args.weights) and os.path.exists('models/g.npz'):
        args.weights = 'models/g.npz'

    run_sr(args.image, args.weights, args.output_dir)
