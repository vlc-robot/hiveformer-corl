#!/bin/bash

#SBATCH --partition=russ_reserved,kate_reserved
#SBATCH --job-name=analogical_manipulation
#SBATCH --output=slurm_logs/train-hiveformer-%j.out
#SBATCH --error=slurm_logs/train-hiveformer-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint=volta|A100
#SBATCH --exclude=matrix-0-24

python train.py "$@"
