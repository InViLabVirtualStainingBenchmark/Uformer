#!/bin/bash
#SBATCH --job-name=infer_uformer_MIST_ER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/infer_uformer_MIST_ER_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/infer_uformer_MIST_ER_%j.err

set -euo pipefail

SHARED=/scratch/antwerpen/grp/ap_invilab_td_thesis
SQSH=${SHARED}/MIST_ER_neutral.sqsh
CONTAINER=${VSC_SCRATCH}/containers/uformer_nvidia.sif
CODE_DIR=${VSC_DATA}/projects/code
OUTPUT_BASE=${SHARED}/benchmark_inference

WEIGHTS=${VSC_DATA}/projects/outputs/uformer_MIST_ER_512_24ep_leibniz/checkpoints/denoising/MIST_ER/Uformer_B_512_24ep/models/model_best.pth

echo "========================================"
echo " Job     : ${SLURM_JOB_ID}"
echo " Model   : Uformer"
echo " Stain   : ER"
echo " Weights : ${WEIGHTS}"
echo "========================================"

srun apptainer exec \
    --nv \
    -B ${SQSH}:/data:image-src=/ \
    -B ${CODE_DIR}:/code \
    -B ${OUTPUT_BASE}:/output \
    -B ${VSC_DATA}/projects/outputs:/outputs \
    --env VSC_DATA=${VSC_DATA} \
    ${CONTAINER} \
    python3 /code/benchmark_inference.py \
        --model   uformer \
        --dataset MIST_ER \
        --weights /outputs/uformer_MIST_ER_512_24ep_leibniz/checkpoints/denoising/MIST_ER/Uformer_B_512_24ep/models/model_best.pth \
        --output_base /output

echo "Done — results at: ${OUTPUT_BASE}/uformer_MIST_ER"
