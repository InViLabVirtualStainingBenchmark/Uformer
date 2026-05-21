
#!/bin/bash

NEPOCH=${1:-13}

RESUME_FLAG=""

PRETRAIN_PATH=""

if [ "$2" == "--resume" ]; then

    RESUME_FLAG="--resume"

    PRETRAIN_PATH="--pretrain_weights $3"

fi

cd /code

nvidia-smi

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

    --nepoch      $NEPOCH \

    --train_ps    512 \

    --val_ps      512 \

    --checkpoint  26 \

    --step_lr     13 \

    --train_workers 4 \

    --eval_workers  4 \

    $RESUME_FLAG \

    $PRETRAIN_PATH \

    2>&1 | tee /output/train_log_ep${NEPOCH}.txt

