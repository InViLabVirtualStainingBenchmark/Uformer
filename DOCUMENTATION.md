** UFORMER BCI HE→IHC **
---

````markdown
# Uformer BCI HE→IHC Virtual Staining Project

## 1. Project Goal

This project adapts the **Uformer architecture** (originally designed for image denoising) to perform **virtual staining**:

- **Input:** H&E (HE) histology tiles  
- **Output:** Immunohistochemistry (IHC) tiles  

The model was trained and evaluated using the **BCI dataset**, within a unified environment shared across multiple virtual staining models.

---

## 2. Environment Setup

We used a unified conda environment:

```bash
conda activate vs_ua
````

This environment includes dependencies for:

* Uformer
* Restormer
* SwinIR
* Other virtual staining models

This ensures compatibility and consistent package versions across repositories.

---

## 3. Dataset Preparation

Dataset directory:

```bash
~/virtual_stain/data/BCI_Uformer/
```

Structure:

```
BCI_Uformer/
    train/
        input/         # HE tiles
        groundtruth/   # IHC tiles
    test/
        input/
        groundtruth/
```

Each HE/IHC pair shares the same filename:

```
00001_train_1+.png
00001_train_1+.png
```

---

## 4. Training the Uformer Model

Training command:

```bash
python train_denoise.py \
    --train_dir ~/virtual_stain/data/BCI_Uformer/train \
    --val_dir   ~/virtual_stain/data/BCI_Uformer/test \
    --arch Uformer_B \
    --batch_size 1 \
    --train_ps 128 \
    --embed_dim 32 \
    --gpu 0 \
    --exp_name Uformer_BCI_HE2IHC
```

### Output Directory

```
~/virtual_stain/outputs/Uformer_BCI_HE2IHC/denoising/SIDD/Uformer_BBCI_HE2IHC_UformerB/
```

Contents:

```
models/
    model_best.pth
    model_latest.pth
logs/
results/
```

---

## 5. Modifications to Uformer

No core Uformer files were modified.

### Custom Additions

#### ✔ Custom Test Script

Created:

```
~/virtual_stain/scripts/test_uformer_bci.py
```

### Features:

* Loads PNG HE images
* Runs Uformer inference
* Saves predicted IHC images
* Computes:

  * PSNR (Peak Signal-to-Noise Ratio)
  * SSIM (Structural Similarity Index)
* Generates labeled side-by-side comparisons

#### ✔ Additional Enhancements

* `--max_images` argument (limit number of test samples)
* Side-by-side visualization:

  * HE Input
  * Predicted IHC
  * Ground Truth IHC
* Automatic padding to multiples of 128 (required for Uformer)
* Fully compatible with unified environment

---

## 6. Running Inference

Navigate to Uformer repository:

```bash
cd ~/virtual_stain/repos/Uformer
```

Run the test script:

```bash
python ../../scripts/test_uformer_bci.py \
    --input_dir  ~/virtual_stain/data/BCI_Uformer/test/input \
    --gt_dir     ~/virtual_stain/data/BCI_Uformer/test/groundtruth \
    --result_dir ~/virtual_stain/outputs/Uformer_BCI_HE2IHC/results \
    --weights    ~/virtual_stain/outputs/Uformer_BCI_HE2IHC/denoising/SIDD/Uformer_BBCI_HE2IHC_UformerB/models/model_best.pth \
    --gpu        0 \
    --max_images 30
```

---

## 7. Output Structure

Results directory:

```
~/virtual_stain/outputs/Uformer_BCI_HE2IHC/results/
```

### Predicted Images

```
predicted_IHC/
    00000_test_1+.png
    00001_test_2+.png
```

### Comparison Visualizations

```
comparison/
    00000_test_1+.png
    00001_test_2+.png
```

Each comparison image contains:

```
[ HE Input | Predicted IHC | Ground Truth IHC ]
```

With labels displayed at the top.

---

## 8. Evaluation Metrics

The script computes:

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

### Example Output

```
Average PSNR : 22.0656 dB
Average SSIM : 0.6288
Images tested: 30
```

---

## 9. Summary of Changes

### Created

```
~/virtual_stain/scripts/test_uformer_bci.py
```

(Custom inference + evaluation script)

### Edited

* None of the core Uformer files

### Used

* `train_denoise.py` (training)
* `utils.get_arch()` (model loading)
* `model_best.pth` (trained weights)

```
