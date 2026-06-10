#!/bin/bash
#SBATCH --job-name=uformer_MIST_PR_leib_512_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p arcturus_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/uformer_MIST_PR_512_24ep_leibniz_p1_leib_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/uformer_MIST_PR_512_24ep_leibniz_p1_leib_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/uformer_rocm.sif"
CODE_DIR="$VSC_DATA/projects/code/Uformer"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_PR_neutral.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/uformer_MIST_PR_512_24ep_leibniz"

mkdir -p "$OUTPUT_DIR"

# nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \

srun apptainer exec \
    --rocm \
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
    bash /code/hpc/train_uformer_MIST_PR.sh 12
    "

