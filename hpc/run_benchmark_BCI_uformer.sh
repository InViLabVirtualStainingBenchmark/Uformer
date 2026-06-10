#!/bin/bash
#SBATCH --job-name=infer_uformer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/infer_uformer_BCI_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/infer_uformer_BCI_%j.err

set -euo pipefail

SHARED=/scratch/antwerpen/grp/ap_invilab_td_thesis
BCI_SQSH=${SHARED}/BCI.sqsh
CONTAINER=${VSC_SCRATCH}/containers/uformer_nvidia.sif
CODE_DIR=${VSC_DATA}/projects/code
OUTPUT_BASE=${VSC_SCRATCH}/benchmark_inference

mkdir -p ${OUTPUT_BASE}

echo "========================================"
echo " Job     : ${SLURM_JOB_ID}"
echo " Model   : Uformer"
echo " Dataset : BCI"
echo " Node    : $(hostname)"
echo "========================================"

srun apptainer exec \
    --nv \
    -B ${BCI_SQSH}:/data:image-src=/ \
    -B ${CODE_DIR}:/code \
    -B ${OUTPUT_BASE}:/output \
    -B ${VSC_DATA}/projects/outputs:/outputs \
    --env VSC_DATA=${VSC_DATA} \
    ${CONTAINER} \
    python3 /code/benchmark_inference.py \
        --model       uformer \
        --dataset     BCI \
        --output_base /output

echo "Done — results at: ${OUTPUT_BASE}/uformer_BCI"
