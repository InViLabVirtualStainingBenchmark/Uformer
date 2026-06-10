"""
Test script for Uformer trained on BCI HE->IHC virtual staining.
Loads HE test images, runs inference, saves predicted IHC images,
and computes PSNR and SSIM against real IHC ground truth.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# Ensure Uformer repo is on the path
dir_name = os.path.dirname(os.path.abspath(__file__))
uformer_root = os.path.abspath(os.path.join(dir_name, "..", "repos", "Uformer"))
sys.path.insert(0, uformer_root)
sys.path.insert(0, os.path.join(uformer_root, "dataset"))

from model import Uformer
import utils

# Argument parser
parser = argparse.ArgumentParser(description='Test Uformer on BCI HE->IHC')

parser.add_argument('--input_dir',  required=True, type=str)
parser.add_argument('--gt_dir',     required=True, type=str)
parser.add_argument('--result_dir', required=True, type=str)
parser.add_argument('--weights',    required=True, type=str)
parser.add_argument('--gpu',        default='0', type=str)

parser.add_argument('--arch', default='Uformer_B', type=str)
parser.add_argument('--embed_dim', default=32, type=int)
parser.add_argument('--win_size', default=8, type=int)
parser.add_argument('--token_projection', default='linear', type=str)
parser.add_argument('--token_mlp', default='leff', type=str)
parser.add_argument('--dd_in', default=3, type=int)

parser.add_argument('--vit_dim', default=256, type=int)
parser.add_argument('--vit_depth', default=12, type=int)
parser.add_argument('--vit_nheads', default=8, type=int)
parser.add_argument('--vit_mlp_dim', default=512, type=int)
parser.add_argument('--vit_patch_size', default=16, type=int)
parser.add_argument('--global_skip', action='store_true', default=False)
parser.add_argument('--local_skip', action='store_true', default=False)
parser.add_argument('--vit_share', action='store_true', default=False)
parser.add_argument('--train_ps', default=128, type=int)

parser.add_argument('--max_images', default=None, type=int,
                    help='Limit number of test images')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Output folders
result_dir_img = os.path.join(args.result_dir, 'predicted_IHC')
comparison_dir = os.path.join(args.result_dir, 'comparison')

os.makedirs(result_dir_img, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# Load model
model = utils.get_arch(args)
utils.load_checkpoint(model, args.weights)
print(f"Loaded weights from: {args.weights}")

model.cuda()
model.eval()

# Pad image to multiple of factor
def pad_to_multiple(tensor, factor=128):
    _, _, h, w = tensor.size()
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    if h_pad > 0 or w_pad > 0:
        tensor = F.pad(tensor, (0, w_pad, 0, h_pad), mode='reflect')
    return tensor, h, w

# Collect test images
input_files = sorted([
    f for f in os.listdir(args.input_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
])

if args.max_images is not None:
    input_files = input_files[:args.max_images]

print(f"Found {len(input_files)} test images")

to_tensor = transforms.ToTensor()

psnr_list = []
ssim_list = []

# Inference loop
with torch.no_grad():
    for fname in tqdm(input_files, desc='Testing'):

        input_path = os.path.join(args.input_dir, fname)
        inp_img = Image.open(input_path).convert('RGB')
        inp_tensor = to_tensor(inp_img).unsqueeze(0).cuda()

        inp_padded, orig_h, orig_w = pad_to_multiple(inp_tensor, factor=128)

        output = model(inp_padded)
        output = output[:, :, :orig_h, :orig_w]
        output = torch.clamp(output, 0, 1)

        pred_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred_np = img_as_ubyte(pred_np)

        # Save predicted IHC
        save_path = os.path.join(result_dir_img, fname)
        Image.fromarray(pred_np).save(save_path)

        # Metrics
        gt_path = os.path.join(args.gt_dir, fname)
        if os.path.exists(gt_path):
            gt_img = np.array(Image.open(gt_path).convert('RGB'))

            psnr_val = psnr_loss(gt_img, pred_np, data_range=255)
            ssim_val = ssim_loss(gt_img, pred_np, data_range=255, channel_axis=2)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

        # SIDE-BY-SIDE WITH LABELS
        he_np = np.array(inp_img)

        if os.path.exists(gt_path):
            gt_np = gt_img
            canvas = np.concatenate([he_np, pred_np, gt_np], axis=1)
            labels = ["HE Input", "Predicted IHC", "Ground Truth IHC"]
        else:
            canvas = np.concatenate([he_np, pred_np], axis=1)
            labels = ["HE Input", "Predicted IHC"]

        canvas_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas_img)

        # Try to load a font; fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

        w = he_np.shape[1]

        for i, label in enumerate(labels):
            draw.text((i * w + 10, 10), label, fill=(255, 0, 0), font=font)

        comp_path = os.path.join(comparison_dir, fname)
        canvas_img.save(comp_path)

# Final results
print("\n" + "="*50)
print("Uformer BCI HE->IHC Test Results")
print("="*50)
if psnr_list:
    print(f"Average PSNR : {np.mean(psnr_list):.4f} dB")
    print(f"Average SSIM : {np.mean(ssim_list):.4f}")
    print(f"Images tested: {len(psnr_list)}")
else:
    print("No ground truth found — images saved without metrics")

print(f"\nPredicted images saved to: {result_dir_img}")
print(f"Comparison images saved to: {comparison_dir}")
