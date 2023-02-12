#!/bin/bash

#SBATCH --partition=russ_reserved,kate_reserved
#SBATCH --job-name=train_hiveformer
#SBATCH --output=slurm_logs/train-hiveformer-%j.out
#SBATCH --error=slurm_logs/train-hiveformer-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=100gb
#SBATCH --mem-per-gpu=32gb

python train.py "$@"
