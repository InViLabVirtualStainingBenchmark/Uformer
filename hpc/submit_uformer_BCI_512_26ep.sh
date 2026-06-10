#!/bin/bash
JOB1=$(sbatch --parsable $VSC_DATA/projects/jobs/train_uformer_BCI_512_26ep_part1.sh)
echo "Submitted Uformer BCI Part 1: job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
  $VSC_DATA/projects/jobs/train_uformer_BCI_512_26ep_part2.sh)
echo "Submitted Uformer BCI Part 2: job $JOB2 (starts after $JOB1)"
