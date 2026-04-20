# Uformer: Image Denoising Pipeline
A Complete Workflow for Modern GPU Systems (RTX 4080)
---
## 1. Overview
 
This repository implements **Uformer**, a transformer-based image restoration model designed for tasks such as:
 
- Image denoising
- Deblurring
- Deraining
- Low-light enhancement
 
This README documents the full workflow used to run Uformer on the **SIDD** dataset, including environment setup, dependency fixes, inference, and visualization. A dedicated section explains how Uformer can be applied to **pathology datasets**.
 
---
 
## 2. Hardware & Environment
 
### 2.1 Hardware Used
- Ubuntu 22.04
- NVIDIA RTX 4080 (16GB VRAM)
- Python 3.10
- Conda environment: `uformer_gpu`
 
### 2.2 Required Environment
 
The original Uformer repository was built for older versions of PyTorch and NumPy. To run Uformer on modern GPUs (e.g., RTX 4080), we created a clean, compatible environment.
 
### 2.3 Environment Reconstruction
 
```bash
conda create -n uformer_gpu python=3.10 -y
conda activate uformer_gpu
```
 
Install PyTorch with CUDA 12.1 (required for RTX 4080):
 
```bash# 
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```
 
Install Uformer dependencies:
 
```bash
pip install einops timm scikit-image opencv-python scipy tqdm h5py matplotlib natsort
```
 
### 2.4 NumPy Compatibility Fix
 
PyTorch 2.2.0 is not compatible with NumPy 2.x, so we downgraded:
 
```bash
pip install "numpy<2"
```
 
**Why this change was needed:**
NumPy 2.x breaks PyTorch's C-extensions, causing `RuntimeError: Numpy is not available`. Downgrading restores full GPU functionality.
 
---
 
## 3. Dataset Structure (SIDD)
 
Uformer expects the SIDD validation dataset in the following structure:
 
```
datasets/denoising/sidd/val/
   ├── input/
   ├── groundtruth/
   ├── ValidationNoisyBlocksSrgb.mat
   └── ValidationGtBlocksSrgb.mat
```
 
### 3.1 What Uformer Uses
 
The inference script (`test_sidd.py`) loads:
 
- Noisy patches from `ValidationNoisyBlocksSrgb.mat`
- Ground truth patches from `ValidationGtBlocksSrgb.mat`
 
The PNGs inside `input/` are not used by the model, but they are useful for visualization.
 
---
 
## 4. Running Inference
 
From the Uformer root directory:
 
```bash
python test/test_sidd.py \
 --input_dir datasets/denoising/sidd/val \
 --result_dir results/sidd \
 --weights checkpoints/denoising/Uformer_B.pth \
 --gpus 0
```
 
This produces:
 
```
results/sidd/png/   → denoised PNG patches
results/sidd/mat/   → denoised .mat patches
```
 
---
 
## 5. Viewing Results
 
A simple viewer script (`view_sidd_result.py`) displays noisy vs. denoised patches:
 
```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
 
noisy_dir = "datasets/denoising/sidd/val/input"
denoised_dir = "results/sidd/png"
 
patch = "0001_01.png"
 
noisy = np.array(Image.open(os.path.join(noisy_dir, patch)))
denoised = np.array(Image.open(os.path.join(denoised_dir, patch)))
 
plt.figure(figsize=(12, 6))
 
plt.subplot(1, 2, 1)
plt.title("Noisy Input")
plt.imshow(noisy)
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.title("Denoised (Uformer)")
plt.imshow(denoised)
plt.axis("off")
 
plt.show()
```
 
---
 
## 6. What We Changed & Why
 
- ✔ **Created a new environment** — The original repo was incompatible with modern GPUs and Python versions.
- ✔ **Installed PyTorch 2.2.0 + CUDA 12.1** — Required for RTX 4080 support.
- ✔ **Downgraded NumPy to < 2** — Necessary because PyTorch 2.2.0 cannot load NumPy 2.x.
- ✔ **Installed missing dependencies** — `einops`, `timm`, `natsort`, `opencv-python`, etc.
- ✔ **Adjusted viewer script** — Because SIDD stores noisy images in `.mat` files, not PNGs.
- ✔ **Verified dataset structure** — Ensured Uformer could correctly load SIDD validation patches.
 
---
 
## 7. Use-Case Scenario: Pathology & Medical Imaging
 
Although Uformer was originally designed for natural image restoration, its architecture makes it highly valuable for digital pathology and other biomedical imaging tasks.
 
### 7.1 Why Uformer Works Well for Pathology
 
Pathology images often suffer from:
 
- Scanner noise
- Autofluorescence noise
- Low-light noise (especially fluorescence microscopy)
- Compression artifacts
- Blur from tissue thickness
- Variability in staining intensity
 
Uformer's transformer-based architecture provides:
 
- **Global context awareness** (important for tissue structure)
- **Local detail preservation** (important for nuclei and cell boundaries)
- **Patch-based processing** (matches WSI tiling workflows)
- **Strong denoising performance** on real sensor noise
 
### 7.2 Practical Applications
 
**✔ Preprocessing for Virtual Staining**
Cleaner H&E patches → better performance for models like PSPStain, CycleGAN, or diffusion-based stain translation.
 
**✔ Improving Segmentation Models**
Denoised patches improve nuclei segmentation, gland segmentation, and tumor region detection.
 
**✔ Enhancing Fluorescence Microscopy**
Uformer can remove shot noise, sensor noise, and low-light noise, improving downstream quantification.
 
**✔ Stabilizing ML Pipelines**
Noise reduction improves feature extraction, embedding consistency, and classification accuracy.
 
### 7.3 Why It Matters
 
In pathology, signal quality is everything. Cleaner images → better models → more reliable diagnostics. Uformer provides a fast, transformer-based denoising solution that integrates seamlessly into modern digital pathology pipelines.
 
---
This repo wasnt tested on neither BCI OR MINST because it is a denoising model not a virtual staining model

---
 
## 8. Summary
 
This documentation outlines:
 
- A fully working Uformer environment for RTX 4080
- Dataset preparation for SIDD
- Inference and visualization
- Required compatibility fixes
- Pathology-focused use-case justification
 
Uformer is now fully operational on modern hardware and ready for integration into medical imaging workflows.

