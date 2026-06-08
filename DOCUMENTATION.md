--- 
# Uformer: BCI Virtual Staining
Local setup and initial training for H&E → IHC translation · BCI Dataset · Thomas's PC

> **Note:** This documents the local development setup used on Thomas's PC before HPC training.
> For HPC training, inference, and evaluation on the CalcUA cluster, see [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).
> See the [official Uformer repository](https://github.com/ZhendongWang6/Uformer) for the original codebase.

---

## 1. Project Goal

This adapts Uformer (originally designed for image denoising) to perform **virtual histological staining**:

- **Input:** H&E stained histology tiles
- **Output:** IHC stained equivalents

Trained and evaluated on the **BCI dataset** as part of the InViLab Virtual Staining Benchmark.

---

## 2. Environment Setup

A unified conda environment was used across all models in the benchmark:

```bash
conda activate vs_ua
```

This environment covers dependencies for Uformer, Restormer, SwinIR, and other models in the benchmark.

> **To do:** verify exact packages installed in `vs_ua` on Thomas's PC and document them here.

---

## 3. Dataset Preparation

The BCI dataset was structured to match Uformer's expected `input/` and `groundtruth/` folder names.

> **Note:** The preferred approach is to adapt the dataloader to the dataset format, not the other way around. The dataloader (`dataset/dataset_denoise.py`) expects `input/` and `groundtruth/` subdirectories. For HPC training this was solved with runtime symlinks — see [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).

Dataset directory on Thomas's PC:

```
~/virtual_stain/data/BCI_Uformer/
├── train/
│   ├── input/         ← H&E tiles
│   └── groundtruth/   ← IHC tiles
└── test/
    ├── input/
    └── groundtruth/
```

Each H&E and IHC pair shares the same filename:

```
00001_train_1+.png  ← H&E
00001_train_1+.png  ← IHC (same name, different folder)
```

> **Note:** At the time of local training there was no separate validation split — `test/` was used as the validation set. This may affect reproducibility of the training command below.

---

## 4. Training

```bash
python train/train_denoise.py \
    --train_dir ~/virtual_stain/data/BCI_Uformer/train \
    --val_dir   ~/virtual_stain/data/BCI_Uformer/test \
    --arch      Uformer_B \
    --batch_size 1 \
    --train_ps  128 \
    --embed_dim 32 \
    --gpu       0 \
    --dataset   BCI \
    --env       _BCI_HE2IHC \
    --step_lr   13
```

> **To do:** verify this command runs correctly on Thomas's PC with the current repo state.

### Output Structure

The output path is constructed automatically from `--save_dir`, `--dataset`, `--arch`, and `--env`:

```
{save_dir}/denoising/{dataset}/{arch}{env}/
```

For the command above this resolves to:

```
~/virtual_stain/outputs/Uformer_BCI_HE2IHC/denoising/BCI/Uformer_B_BCI_HE2IHC/
├── models/
│   ├── model_best.pth      ← saved when validation PSNR improves
│   └── model_latest.pth    ← saved after every epoch
└── <timestamp>.txt         ← training log
```

> **Why `denoising/` in the path?** This subfolder is hardcoded in `train/train_denoise.py` at the log directory construction line. It can be changed there if needed — look for `log_dir = os.path.join(opt.save_dir, 'denoising', ...)`.

---

## 5. Inference

Inference uses `script/test_uformer_bci.py` — a custom script for BCI virtual staining evaluation.

> ⚠️ **This script is currently empty in the repository.** The working version needs to be recovered from Thomas's PC. See cleanup note #1.

When complete, it will:
- Load full H&E test images
- Run Uformer inference
- Save predicted IHC images
- Compute PSNR and SSIM
- Generate side-by-side H&E / Predicted IHC / Ground Truth comparisons

Expected usage (to be verified):

```bash
python script/test_uformer_bci.py \
    --input_dir  ~/virtual_stain/data/BCI_Uformer/test/input \
    --gt_dir     ~/virtual_stain/data/BCI_Uformer/test/groundtruth \
    --result_dir ~/virtual_stain/outputs/Uformer_BCI_HE2IHC/results \
    --weights    ~/virtual_stain/outputs/.../models/model_best.pth \
    --gpu        0
```

---

## 6. Modifications to Uformer

The following files were changed from the original repository:

### `train/train_denoise.py`

**Line 95** — `step_lr` was hardcoded to 50:

```python
# Original
step = 50

# Modified
step = opt.step_lr
```

This allows the learning rate decay step to be controlled via CLI, which is required for the two-part chained job setup on HPC.

### `utils/image_utils.py`

`is_png_file()` was extended to also accept JPEG files:

```python
# Original
return any(filename.endswith(extension) for extension in [".png"])

# Modified
return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
```

This was needed for MIST dataset support — MIST images are JPEGs, not PNGs.

---

## 7. Results

Initial local run (30 test images, `train_ps 128`):

```
Average PSNR : 22.0656 dB
Average SSIM : 0.6288
Images tested: 30
```

> Full benchmark results (512×512 patch, 26 epochs, HPC training) are in [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).


