#!/bin/bash
#SBATCH --job-name=uformer_MIST_HER2_leib_512_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/uformer_MIST_HER2_512_24ep_leibniz_p2_leib_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/uformer_MIST_HER2_512_24ep_leibniz_p2_leib_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/uformer_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code/Uformer"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_HER2_neutral.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/uformer_MIST_HER2_512_24ep_leibniz"
CHECKPOINT="/output/checkpoints/denoising/MIST_HER2/Uformer_B_512_24ep/models/model_latest.pth"

mkdir -p "$OUTPUT_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 > "$OUTPUT_DIR/gpu_usage_p2.csv" &
GPU_LOG_PID=$!

srun apptainer exec \
    --nv \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$OUTPUT_DIR":/output \
    "$CONTAINER" \
    bash -c "
    mkdir -p /tmp/mist/train /tmp/mist/val
    ln -s /data/train/HE  /tmp/mist/train/input
    ln -s /data/train/IHC /tmp/mist/train/groundtruth
    ln -s /data/val/HE    /tmp/mist/val/input
    ln -s /data/val/IHC   /tmp/mist/val/groundtruth
    bash /code/hpc/train_uformer_MIST_HER2.sh 24 --resume $CHECKPOINT
    "

kill $GPU_LOG_PID || true
