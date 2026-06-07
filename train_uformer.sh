#!/bin/bash
NEPOCH=${1:-307}
cd /code
nvidia-smi
python3 train/train_denoise.py \
    --arch Uformer_B \
    --batch_size 8 \
    --gpu 0 \
    --train_dir /data/train \
    --val_dir /data/val \
    --embed_dim 32 \
    --save_dir /output/checkpoints \
    --nepoch $NEPOCH \
    --train_ps 256 \
    --train_workers 4 \
    --eval_workers 4 \
    2>&1 | tee /output/train_log.txt
