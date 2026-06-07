
# Uformer — Virtual Staining Benchmark
### H&E → IHC Translation · BCI & MIST Datasets · CalcUA HPC (Vaughan A100)
 
---
 
> This repository adapts **Uformer** (U-shaped Transformer for image restoration) for virtual staining — translating H&E histology images to IHC-stained equivalents. It is part of the **InViLab Virtual Staining Benchmark** comparing transformer-based image restoration models across two datasets: BCI and MIST.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Environment](#environment)
- [Repository Structure](#repository-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Modifications](#modifications)
- [Notes](#notes)
---
 
## Overview
 
Uformer uses a U-Net style hierarchical transformer architecture for image restoration. In this benchmark it is applied to **H&E → IHC virtual staining** using pixel-level supervised training with a Charbonnier loss.
 
**Datasets:**
| Dataset | Task | Train | Val | Test |
|---------|------|-------|-----|------|
| BCI | H&E → IHC | 3896 | 488 | 489 |
| MIST ER | H&E → ER IHC | 4153 | 500 | 500 |
| MIST HER2 | H&E → HER2 IHC | 4642 | 500 | 500 |
| MIST Ki67 | H&E → Ki67 IHC | 4361 | 500 | 500 |
| MIST PR | H&E → PR IHC | 4139 | 500 | 500 |
 
**Key training settings:**
| Parameter | Value |
|-----------|-------|
| Architecture | Uformer_B |
| Input patch size | 512 × 512 |
| Embed dim | 32 |
| Batch size | 1 |
| Total epochs | 26 (BCI) / 24 (MIST) |
| Loss function | Charbonnier |
| Optimizer | AdamW (lr=2e-4) |
| LR scheduler | StepLR (step=13 for BCI, step=12 for MIST) |
 
---
 
## Environment
 
Training runs inside a Singularity/Apptainer container on the **CalcUA Vaughan cluster** (NVIDIA A100 40GB, `ampere_gpu` partition).
 
**Container:** `uformer_nvidia.sif`
- Base image: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`
- PyTorch: 1.13.1+cu116
- Python: 3.9
**Container location on cluster:**
```
$VSC_SCRATCH/containers/uformer_nvidia.sif
```
 
---
 
## Repository Structure
 
```
Uformer/
├── train/
│   └── train_denoise.py        ← Main training script (modified for virtual staining)
├── dataset/
│   └── dataset_denoise.py      ← Dataset loader
│       ├── DataLoaderTrain      ← Crops to train_ps during training
│       └── DataLoaderVal        ← Full images during validation (no crop)
├── utils/
│   └── model_utils.py          ← load_checkpoint, load_start_epoch, load_optim
├── model.py                    ← Uformer architecture
├── options.py                  ← All CLI arguments
├── losses.py                   ← Charbonnier loss
├── train_uformer_BCI.sh        ← BCI training wrapper
├── train_uformer_MIST_ER.sh    ← MIST ER training wrapper
├── train_uformer_MIST_HER2.sh
├── train_uformer_MIST_Ki67.sh
├── train_uformer_MIST_PR.sh
└── test_uformer_bci.py         ← Custom inference script
```
 
---
 
## Dataset Preparation
 
### Squashfs Format
 
All datasets are stored as **SquashFS images** (`.sqsh`) for fast HPC I/O. Each squashfs uses a **neutral folder structure**:
 
```
dataset.sqsh (mounted at /data)
├── train/
│   ├── HE/        ← H&E input images
│   └── IHC/       ← IHC ground truth images
├── val/
│   ├── HE/
│   └── IHC/
└── test/
    ├── HE/
    └── IHC/
```
 
**Squashfs locations (shared group storage):**
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_ER_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_HER2_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_Ki67_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_PR_neutral.sqsh
```
 
### Runtime Symlinks
 
Uformer's dataset loader (`DataLoaderTrain`) expects this structure:
 
```
train_dir/
    input/         ← H&E images
    groundtruth/   ← IHC images
```
 
Since the neutral squashfs uses `train/HE/` and `train/IHC/`, the job scripts create symlinks at runtime:
 
```bash
mkdir -p /tmp/bci/train /tmp/bci/val
ln -s /data/train/HE  /tmp/bci/train/input
ln -s /data/train/IHC /tmp/bci/train/groundtruth
ln -s /data/val/HE    /tmp/bci/val/input
ln -s /data/val/IHC   /tmp/bci/val/groundtruth
```
 
The training wrappers then pass `--train_dir /tmp/bci/train --val_dir /tmp/bci/val`.
 
> `[ADD IMAGE: diagram showing squashfs neutral format → symlinks → Uformer input/groundtruth structure]`
 
---
 
## Training
 
### How Training Works
 
Training is split into **two chained SLURM jobs** because a full run exceeds the 23-hour wall time limit at 512×512 patch size.
 
**BCI (26 epochs total):**
- Part 1: epochs 1 → 13
- Part 2: resumes from `model_latest.pth` at epoch 13, continues to epoch 26
**MIST (24 epochs total):**
- Part 1: epochs 1 → 12
- Part 2: resumes from `model_latest.pth` at epoch 12, continues to epoch 24
Resuming uses `--resume` and `--pretrain_weights` pointing to the `model_latest.pth` saved by part 1.
 
### Epoch Count Rationale
 
Epoch counts were chosen to match approximately 100k training iterations for fair comparison with NAFNet:
 
| Dataset | Images | Iters/epoch | Target iters | Epochs |
|---------|--------|-------------|--------------|--------|
| BCI | 3896 | 3896 | ~100k | 26 |
| MIST ER | 4153 | 4153 | ~100k | 24 |
 
### Training Wrappers
 
Each wrapper sets the correct `--train_dir`, `--val_dir`, dataset name, and environment tag:
 
**BCI** (`train_uformer_BCI.sh`):
```bash
python3 train/train_denoise.py \
    --arch        Uformer_B \
    --batch_size  1 \
    --gpu         0 \
    --train_dir   /tmp/bci/train \
    --val_dir     /tmp/bci/val \
    --embed_dim   32 \
    --save_dir    /output/checkpoints \
    --dataset     BCI \
    --env         _512_26ep \
    --nepoch      <13 or 26> \
    --train_ps    512 \
    --checkpoint  26 \
    --step_lr     13 \
    ...
```
 
### Job Scripts
 
```
$VSC_DATA/projects/jobs/
├── train_uformer_BCI_512_26ep_part1.sh
├── train_uformer_BCI_512_26ep_part2.sh
├── submit_uformer_BCI_512_26ep.sh
├── train_uformer_MIST_ER_512_24ep_part1.sh
├── train_uformer_MIST_ER_512_24ep_part2.sh
├── submit_uformer_MIST_ER_512_24ep.sh
└── ... (HER2, Ki67, PR)
```
 
### Submitting Training
 
**BCI:**
```bash
bash $VSC_DATA/projects/jobs/submit_uformer_BCI_512_26ep.sh
```
 
**MIST (all 4 biomarkers):**
```bash
for marker in ER HER2 Ki67 PR; do
    bash $VSC_DATA/projects/jobs/submit_uformer_MIST_${marker}_512_24ep.sh
done
```
 
### Output Structure
 
```
$VSC_DATA/projects/outputs/uformer_BCI_512_26ep/
└── checkpoints/
    └── denoising/
        └── BCI/
            └── Uformer_B_512_26ep/
                ├── models/
                │   ├── model_best.pth      ← Best validation PSNR
                │   ├── model_latest.pth    ← Saved after every epoch (used to resume)
                │   └── model_epoch_26.pth  ← Final checkpoint
                ├── results/
                └── <timestamp>.txt         ← Training log
```
 
### Monitoring
 
```bash
# Check running jobs
squeue -u vsc21216
 
# Watch training progress (Uformer logs every quarter-epoch)
tail -f $VSC_DATA/projects/logs/uformer_BCI_512_26ep_p1_<JOBID>.out
```
 
---
 
## Inference
 
Inference uses the custom `test_uformer_bci.py` script which:
- Loads full 1024×1024 test images
- Pads to multiples of 128 (required by Uformer's window attention)
- Runs inference
- Saves predicted IHC, ground truth IHC, and H&E input
- Computes PSNR and SSIM
> `[ADD IMAGE: side-by-side example of HE input, predicted IHC, ground truth IHC]`
 
---
 
## Evaluation
 
Evaluation uses the shared `evaluate.py` script from the InViLab benchmark repository, run inside the `evaluate_nvidia.sif` container on the `broadwell` (CPU) partition of Leibniz.
 
**Metrics computed:**
- PSNR, SSIM, MS-SSIM
- LPIPS (AlexNet and VGG)
- MAE
- FID
- Cellpose cell detection F1 (optional, `--cellpose`)
**Results are appended to:**
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_results.csv
```
 
---
 
## Results
 
### BCI Dataset
 
| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|-----------|--------------|-------------|-------|-------|
| Uformer (512 crop, 26 epochs) | — | — | — | — | — | — | — |
 
*Results will be updated after training completes.*
 
### MIST Dataset
 
| Model | Marker | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | FID ↓ |
|-------|--------|--------|--------|--------------|-------|
| Uformer | ER | — | — | — | — |
| Uformer | HER2 | — | — | — | — |
| Uformer | Ki67 | — | — | — | — |
| Uformer | PR | — | — | — | — |
 
---
 
## Modifications
 
The following files were modified from the original Uformer repository:
 
### `train/train_denoise.py`
 
**Line 95** — `step_lr` was hardcoded to 50. Changed to use the CLI argument:
 
```python
# Before
step = 50
 
# After
step = opt.step_lr
```
 
This allows proper learning rate decay scheduling across the two-part job chain (decay at epoch 13 for BCI, epoch 12 for MIST).
 
---
 
## Notes
 
- **Validation uses full images** — `DataLoaderVal` loads images without any cropping. Only training applies the 512×512 patch crop via `DataLoaderTrain`.
- **`val_ps` argument is not used** — the validation dataloader ignores `--val_ps`. Validation always runs on full images regardless of this setting.
- **`model_latest.pth` is saved after every epoch** — this is what part 2 resumes from. It stores the epoch number, model weights, and optimizer state so the learning rate schedule continues correctly.
- **`model_best.pth` is saved whenever validation PSNR improves** — use this checkpoint for final inference, not `model_latest.pth`.
- **Always use the neutral squashfs** (`BCI.sqsh`, `MIST_*_neutral.sqsh`) with the runtime symlink pattern. The old `BCI_Uformer_split.sqsh` and `MIST_*_Uformer.sqsh` files have been deleted.
- **The `--dataset` and `--env` arguments** determine the output folder path: `{save_dir}/denoising/{dataset}/Uformer_B{env}/`. For BCI this is `denoising/BCI/Uformer_B_512_26ep/`.
