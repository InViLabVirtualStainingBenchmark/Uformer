#!/bin/bash
JOB1=$(sbatch --parsable /data/antwerpen/212/vsc21216/projects/jobs/train_uformer_MIST_Ki67_512_24ep_part1.sh)
echo "Submitted Uformer MIST Ki67 Part 1: job $JOB1"
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
  /data/antwerpen/212/vsc21216/projects/jobs/train_uformer_MIST_Ki67_512_24ep_part2.sh)
echo "Submitted Uformer MIST Ki67 Part 2: job $JOB2 (starts after $JOB1)"
