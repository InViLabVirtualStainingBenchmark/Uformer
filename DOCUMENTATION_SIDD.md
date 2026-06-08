# Uformer: SIDD Denoising Pipeline
Local setup and inference on the SIDD dataset · RTX 4080 · Ubuntu 22.04

> **Note:** This documents the setup used on the development machine (Thomas's PC).
> The conda environment name `uformer_gpu` was used during development — you can use any name.
> See the [official Uformer repository](https://github.com/ZhendongWang6/Uformer) for the original codebase.

---

## 1. Hardware & Environment

- **OS:** Ubuntu 22.04
- **GPU:** NVIDIA RTX 4080 (16GB VRAM)
- **Python:** 3.10
- **Conda environment:** `uformer_gpu` (development name — rename as needed)

---

## 2. Environment Setup

The original Uformer `requirements.txt` specifies `torch==1.8.0` which is incompatible with modern GPUs like the RTX 4080. Do not use it directly. Use the following setup instead, which was verified to work on RTX 4080 hardware.

```bash
conda create -n uformer_gpu python=3.10 -y
conda activate uformer_gpu
```

Install PyTorch with CUDA 12.1:

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies:

```bash
pip install einops timm scikit-image opencv-python scipy tqdm h5py matplotlib natsort
```

Downgrade NumPy (PyTorch 2.2.0 is incompatible with NumPy 2.x):

```bash
pip install "numpy<2"
```

> **To do:** verify the full list of packages installed in `uformer_gpu` on Thomas's PC and update this section if anything is missing.

---

## 3. Dataset Structure

Download the SIDD validation dataset from the [official SIDD page](https://mailustceducn-my.sharepoint.com/personal/zhendongwang_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhendongwang%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FUformer%2Fdatasets&ga=1) and place it as follows:

```
datasets/denoising/sidd/val/
├── input/                          ← noisy PNG patches (visualization only)
├── groundtruth/                    ← clean PNG patches (visualization only)
├── ValidationNoisyBlocksSrgb.mat   ← used by test/test_sidd.py
└── ValidationGtBlocksSrgb.mat      ← used by test/test_sidd.py
```

> `test/test_sidd.py` loads patches from the `.mat` files, not the PNGs. The PNGs are only used by the viewer script.

---

## 4. Running Inference

Download the pretrained `Uformer_B.pth` weights from the [official repo](https://github.com/ZhendongWang6/Uformer) and place at `checkpoints/denoising/Uformer_B.pth`.

From the Uformer root directory:

```bash
python test/test_sidd.py \
    --input_dir datasets/denoising/sidd/val \
    --result_dir results/sidd \
    --weights checkpoints/denoising/Uformer_B.pth \
    --gpus 0
```

> `--gpus 0` selects the first GPU (device index 0). Change to `--gpus 1` for the second GPU if available.

Output:

```
results/sidd/png/   → denoised PNG patches
results/sidd/mat/   → denoised .mat patches
```

---

## 5. Viewing Results

A viewer script (`view_sidd_result.py`) is provided in the root directory. It displays noisy input vs denoised output side by side.

**Show one random patch:**
```bash
python view_sidd_result.py
```

**Show a specific patch:**
```bash
python view_sidd_result.py --patch 0001_01.png
```

**Show a grid of 4 random patches:**
```bash
python view_sidd_result.py --num_patches 4
```

**Save the figure instead of displaying:**
```bash
python view_sidd_result.py --num_patches 4 --save --output_dir results/figures
```

---

## 6. Changes Made

| What | Why |
|------|-----|
| Created new conda environment `uformer_gpu` | Original repo incompatible with modern GPUs and Python versions |
| PyTorch 2.2.0 + CUDA 12.1 | Required for RTX 4080 support |
| NumPy downgraded to `<2` | NumPy 2.x breaks PyTorch C-extensions (`RuntimeError: Numpy is not available`) |
| Created `view_sidd_result.py` | Easy viewing of results after inference — supports CLI args and multi-patch grid |

---

## 7. Pathology Context

Uformer was originally designed for natural image restoration. Its transformer-based architecture also makes it suitable for digital pathology:

- **Global context awareness** — captures tissue-level structure
- **Local detail preservation** — maintains nuclei and cell boundaries
- **Patch-based processing** — compatible with WSI tiling workflows
```

