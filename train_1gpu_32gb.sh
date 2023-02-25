#!/bin/bash

#SBATCH --partition=russ_reserved,kate_reserved
#SBATCH --job-name=analogical-manip
#SBATCH --output=slurm_logs/analogical-manip-%j.out
#SBATCH --error=slurm_logs/analogical-manip-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50gb
#SBATCH --constraint=volta|A100
#SBATCH --exclude=matrix-0-24

python train.py "$@"
