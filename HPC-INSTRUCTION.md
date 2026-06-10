--- 
# Uformer: HPC Virtual Staining Benchmark
H&E → IHC Translation · BCI & MIST Datasets · CalcUA HPC (Vaughan A100)

> This documents HPC training, inference, and evaluation for Uformer as part of the InViLab Virtual Staining Benchmark. For local setup and initial BCI experiments, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Table of Contents
- [Overview](#overview)
- [Environment](#environment)
- [Cluster Structure](#cluster-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Modifications](#modifications)
- [Notes](#notes)

---

## Overview

Uformer uses a U-Net style hierarchical transformer architecture for image restoration, applied here to H&E → IHC virtual staining using pixel-level supervised training with Charbonnier loss.

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

Training runs inside an Apptainer container on the CalcUA Vaughan cluster.

**Primary container:** `uformer_nvidia.sif` (NVIDIA A100, `ampere_gpu` partition)
- Base image: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`
- PyTorch: 1.13.1+cu116
- Python: 3.9

**ROCm container:** `uformer_rocm.sif` (AMD MI100, `arcturus_gpu` partition)
- Built from `uformer_rocm.def` in `$VSC_SCRATCH/containers/`
- PyTorch: 2.1.2+rocm5.6
- Required for arcturus nodes — `basicsr_rocm.sif` cannot be used for Uformer (missing natsort)

**Container locations:**
```
$VSC_SCRATCH/containers/uformer_nvidia.sif   ← NVIDIA training
$VSC_SCRATCH/containers/uformer_rocm.sif     ← AMD training
```

---

## Cluster Structure

**Compute nodes used:**

| Partition | Node | GPU | Used for |
|-----------|------|-----|----------|
| ampere_gpu | nvam1.vaughan | 4× A100 40GB | BCI training (primary) |
| pascal_gpu | nvpa1/nvpa2.leibniz | 2× P100 16GB | MIST HER2 training |
| arcturus_gpu | amdarc1/amdarc2.vaughan | 2× MI100 32GB | MIST Ki67/PR training |

**Key paths:**
```
$VSC_DATA/projects/code/Uformer/          ← repository
$VSC_DATA/projects/logs/                  ← job logs
$VSC_DATA/projects/outputs/               ← training checkpoints
$VSC_SCRATCH/containers/                  ← Apptainer containers
/scratch/antwerpen/grp/ap_invilab_td_thesis/  ← shared group storage
```

---

## Dataset Preparation

All datasets are stored as **SquashFS images** (`.sqsh`) for fast HPC I/O using a neutral folder structure:

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

**Runtime symlinks:**

Uformer's `DataLoaderTrain` expects `input/` and `groundtruth/` subdirectories. Job scripts create symlinks at runtime:

```bash
mkdir -p /tmp/bci/train /tmp/bci/val
ln -s /data/train/HE  /tmp/bci/train/input
ln -s /data/train/IHC /tmp/bci/train/groundtruth
ln -s /data/val/HE    /tmp/bci/val/input
ln -s /data/val/IHC   /tmp/bci/val/groundtruth
```

---

## Training

### How Training Works

A full training run exceeds the 23-hour wall time limit at 512×512 patch size, so training is split into two chained SLURM jobs:

- **BCI (26 epochs total):** Part 1: epochs 1→13 / Part 2: resumes from `model_latest.pth`, epochs 13→26
- **MIST (24 epochs total):** Part 1: epochs 1→12 / Part 2: resumes from `model_latest.pth`, epochs 12→24

### Epoch Count Rationale

Epoch counts were chosen to match approximately 100k training iterations for fair comparison with NAFNet:

| Dataset | Images | Iters/epoch | Target iters | Epochs |
|---------|--------|-------------|--------------|--------|
| BCI | 3896 | 3896 | ~100k | 26 |
| MIST ER | 4153 | 4153 | ~100k | 24 |
| MIST HER2 | 4642 | 4642 | ~100k | 22 |
| MIST Ki67 | 4361 | 4361 | ~100k | 23 |
| MIST PR | 4139 | 4139 | ~100k | 24 |

### Training Wrappers

The root-level `.sh` scripts (`train_uformer_BCI.sh`, `train_uformer_MIST_ER.sh` etc.) are **inner container scripts** called from inside the Apptainer container. They set dataset-specific arguments and are not submitted directly to SLURM.

Example — BCI (`train_uformer_BCI.sh`):
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
    --nepoch      13 \
    --train_ps    512 \
    --checkpoint  26 \
    --step_lr     13
```

### Job Scripts

```
hpc/
├── submit_uformer_BCI_512_26ep.sh              ← chains part1 + part2
├── train_uformer_BCI_512_26ep_part1.sh
├── train_uformer_BCI_512_26ep_part2.sh
├── submit_uformer_MIST_ER_512_24ep.sh
├── train_uformer_MIST_ER_512_24ep_part1.sh     ← ampere_gpu (Vaughan A100)
├── train_uformer_MIST_ER_512_24ep_part1_leibniz.sh  ← pascal/arcturus fallback
├── train_uformer_MIST_ER_512_24ep_part2.sh
├── train_uformer_MIST_ER_512_24ep_part2_leibniz.sh
├── run_benchmark_BCI_uformer.sh                ← BCI inference
├── run_benchmark_MIST_uformer_ER.sh            ← MIST ER inference
├── eval_uformer_BCI_benchmark.sh               ← BCI evaluation
└── ... (same pattern for HER2, Ki67, PR)
```

The `_leibniz` variants were used when Vaughan `ampere_gpu` was unavailable. They are configured for Leibniz `pascal_gpu` (P100) or `arcturus_gpu` (MI100) depending on availability.

### Submitting Training

**BCI:**
```bash
sbatch hpc/submit_uformer_BCI_512_26ep.sh
```

**MIST (all 4 biomarkers):**
```bash
for marker in ER HER2 Ki67 PR; do
    sbatch hpc/submit_uformer_MIST_${marker}_512_24ep.sh
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
                │   ├── model_best.pth      ← best validation PSNR (use for inference)
                │   └── model_latest.pth    ← saved after every epoch
                └── <timestamp>.txt         ← training log

$VSC_DATA/projects/outputs/uformer_MIST_ER_512_24ep_leibniz/
└── checkpoints/
    └── denoising/
        └── MIST_ER/
            └── Uformer_B_512_24ep/
                ├── models/
                │   ├── model_best.pth
                │   └── model_latest.pth
                └── <timestamp>.txt
```

> MIST output directories have `_leibniz` suffix because part 1 was trained on Leibniz nodes.

### Monitoring

```bash
# Check running jobs
squeue -u vsc21216 --format="%.18i %.35j %.8T %.10M %R"

# Watch training log live
tail -f $VSC_DATA/projects/logs/uformer_BCI_512_26ep_p1_<JOBID>.out

# Check quota
myquota
```

---

## Inference

Benchmark inference uses the unified `benchmark_inference.py` script:

```bash
sbatch hpc/run_benchmark_BCI_uformer.sh
```

Results are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_inference/uformer_BCI/
├── comparison/      ← side-by-side PNGs (HE | predicted | GT)
├── predicted/       ← predicted IHC only
├── metrics.csv      ← per-image PSNR and SSIM
└── summary.txt      ← average PSNR and SSIM
```

> A standalone inference script `test/test_uformer_bci.py` is also available for running inference outside the benchmark pipeline. It includes per-image PSNR/SSIM computation and side-by-side comparison image generation.

---

## Evaluation

```bash
sbatch hpc/eval_uformer_BCI_benchmark.sh
```

Runs on the `broadwell` (CPU) partition of Leibniz inside `evaluate_nvidia.sif`.

**Metrics computed:** PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID

Results are appended to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_results.csv
```

---

## Results

### BCI Dataset

| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|-----------|--------------|-------------|-------|-------|
| Uformer (128px crop, early run) | 22.82 | 0.662 | — | — | — | — | 245.68 |
| Uformer_512 (512px crop, 26ep) | **22.68** | **0.659** | 0.5652 | 0.6364 | 0.6154 | 0.0713 | **213.09** |

> FID improved significantly (245 → 213) with 512×512 patch training despite a marginal PSNR difference.

### MIST Dataset

| Model | Marker | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | FID ↓ |
|-------|--------|--------|--------|--------------|-------|
| Uformer | ER | — | — | — | — |
| Uformer | HER2 | — | — | — | — |
| Uformer | Ki67 | — | — | — | — |
| Uformer | PR | — | — | — | — |

*MIST inference in progress. Results will be updated after evaluation completes.*

---

## Modifications

### `train/train_denoise.py`

Line 95 — `step_lr` was hardcoded to 50:

```python
# Original
step = 50

# Modified
step = opt.step_lr
```

Allows learning rate decay step to be controlled via CLI — required for the two-part chained job setup.

### `utils/image_utils.py`

`is_png_file()` extended to accept JPEG files:

```python
# Original
return any(filename.endswith(extension) for extension in [".png"])

# Modified
return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
```

Required for MIST dataset support — MIST images are JPEGs, not PNGs.

---

## Notes

- **Validation uses full images** — `DataLoaderVal` loads images without cropping. Only training applies the 512×512 patch crop.
- **`val_ps` is not used** — the validation dataloader ignores `--val_ps`. Validation always runs on full images.
- **`model_latest.pth` saves after every epoch** — this is what part 2 resumes from.
- **`model_best.pth` saves when validation PSNR improves** — use this for inference.
- **Output path structure** — determined by `--dataset` and `--env`: `{save_dir}/denoising/{dataset}/Uformer_B{env}/`. The `denoising/` subfolder is hardcoded in `train/train_denoise.py`.
- **Always use neutral squashfs** — `BCI.sqsh`, `MIST_*_neutral.sqsh`. Old format squashfs files have been deleted.
- **nvidia-smi graceful fallback** — training wrappers use `nvidia-smi 2>/dev/null || true` so they work on both NVIDIA and AMD nodes without crashing.