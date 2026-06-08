"""
Test script for Uformer trained on BCI HE->IHC virtual staining.
Adapted for HPC container environment (InViLab benchmark).
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Test Uformer on BCI HE->IHC')
parser.add_argument('--input_dir',  required=True, type=str)
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
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

os.makedirs(args.result_dir, exist_ok=True)

from model import Uformer
import utils

model = utils.get_arch(args)
utils.load_checkpoint(model, args.weights)
print(f"Loaded weights from: {args.weights}")
model.cuda()
model.eval()

def pad_to_multiple(tensor, factor=128):
    _, _, h, w = tensor.size()
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor
    if h_pad > 0 or w_pad > 0:
        tensor = F.pad(tensor, (0, w_pad, 0, h_pad), mode='reflect')
    return tensor, h, w

input_files = sorted([
    f for f in os.listdir(args.input_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
])

print(f"Found {len(input_files)} test images")
to_tensor = transforms.ToTensor()

with torch.no_grad():
    for fname in tqdm(input_files, desc='Testing'):
        inp_img = Image.open(os.path.join(args.input_dir, fname)).convert('RGB')
        inp_tensor = to_tensor(inp_img).unsqueeze(0).cuda()
        inp_padded, orig_h, orig_w = pad_to_multiple(inp_tensor, factor=128)
        output = model(inp_padded)
        output = output[:, :, :orig_h, :orig_w]
        output = torch.clamp(output, 0, 1)
        pred_np = img_as_ubyte(output.squeeze(0).permute(1, 2, 0).cpu().numpy())
        Image.fromarray(pred_np).save(os.path.join(args.result_dir, fname))

print(f"Done! {len(input_files)} images saved to: {args.result_dir}")
