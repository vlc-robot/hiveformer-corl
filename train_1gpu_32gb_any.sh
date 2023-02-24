#!/bin/bash

#SBATCH --partition=russ_reserved,kate_reserved
#SBATCH --job-name=analogical_manipulation
#SBATCH --output=slurm_logs/train-hiveformer-%j.out
#SBATCH --error=slurm_logs/train-hiveformer-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --mem-per-gpu=32gb

python train.py "$@"
