#!/usr/bin/env python3
"""
view_sidd_result.py — Visualize Uformer SIDD denoising results.

Displays side-by-side comparison of noisy input vs denoised output.
Can show a single patch or a grid of multiple patches.
Optionally saves the figure to disk.

Usage:
    python view_sidd_result.py
    python view_sidd_result.py --patch 0001_01.png
    python view_sidd_result.py --num_patches 4
    python view_sidd_result.py --num_patches 6 --save --output_dir results/figures
    python view_sidd_result.py --input_dir path/to/noisy --result_dir path/to/denoised
"""

import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Uformer SIDD denoising results')
    parser.add_argument('--input_dir', default='datasets/denoising/sidd/val/input',
                        help='Directory containing noisy input patches')
    parser.add_argument('--result_dir', default='results/sidd/png',
                        help='Directory containing denoised output patches')
    parser.add_argument('--patch', default=None,
                        help='Specific patch filename (e.g. 0001_01.png). '
                             'Overrides --num_patches if set.')
    parser.add_argument('--num_patches', type=int, default=1,
                        help='Number of random patches to show in a grid (default: 1)')
    parser.add_argument('--save', action='store_true',
                        help='Save figure to disk instead of displaying')
    parser.add_argument('--output_dir', default='results/figures',
                        help='Directory to save figure (used with --save)')
    return parser.parse_args()


def load_patch(input_dir, result_dir, patch_name):
    noisy_path = os.path.join(input_dir, patch_name)
    denoised_path = os.path.join(result_dir, patch_name)
    if not os.path.exists(noisy_path):
        print(f"Noisy patch not found: {noisy_path}")
        return None, None
    if not os.path.exists(denoised_path):
        print(f"Denoised patch not found: {denoised_path}")
        return None, None
    return np.array(Image.open(noisy_path)), np.array(Image.open(denoised_path))


def main():
    args = parse_args()

    # Get available patches
    available = sorted(os.listdir(args.result_dir))
    if not available:
        print(f"No patches found in {args.result_dir}")
        return

    # Determine which patches to show
    if args.patch is not None:
        patches = [args.patch]
    else:
        n = min(args.num_patches, len(available))
        patches = random.sample(available, n)
        print(f"Showing {n} random patch(es): {patches}")

    # Load all patches
    loaded = []
    for p in patches:
        noisy, denoised = load_patch(args.input_dir, args.result_dir, p)
        if noisy is not None:
            loaded.append((p, noisy, denoised))

    if not loaded:
        print("No valid patches to display.")
        return

    # Build grid — 2 columns per patch (noisy | denoised)
    n = len(loaded)
    fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))

    # Handle single patch case (axes is 1D not 2D)
    if n == 1:
        axes = [axes]

    for i, (patch_name, noisy, denoised) in enumerate(loaded):
        axes[i][0].imshow(noisy)
        axes[i][0].set_title(f'Noisy — {patch_name}')
        axes[i][0].axis('off')

        axes[i][1].imshow(denoised)
        axes[i][1].set_title(f'Denoised (Uformer) — {patch_name}')
        axes[i][1].axis('off')

    plt.tight_layout()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        fname = f'comparison_{patches[0]}' if len(patches) == 1 else f'comparison_grid_{n}patches.png'
        out_path = os.path.join(args.output_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()