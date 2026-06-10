#!/bin/bash
#SBATCH --job-name=eval_uformer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab
#SBATCH -p broadwell
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/eval_uformer_BCI_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/eval_uformer_BCI_%j.err
set -euo pipefail
CONTAINER="/scratch/antwerpen/grp/ap_invilab_td_thesis/evaluate_nvidia.sif"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh"
GRP_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis"
srun apptainer exec     -B "$DATA_SQSH":/data:image-src=/     -B "$GRP_DIR":/grp     -B "$VSC_DATA/evaluate":/evaluate     "$CONTAINER" python3 /evaluate/evaluate.py         --pred   /grp/benchmark_inference/uformer_BCI/predicted         --gt     /data/test/IHC         --model_name   uformer_BCI         --dataset_name BCI         --split_name   test         --match_by     stem         --output       /grp/benchmark_results.csv         --device       cpu
