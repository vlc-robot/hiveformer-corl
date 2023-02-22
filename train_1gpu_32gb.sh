#!/bin/bash

#SBATCH --partition=russ_reserved
#SBATCH --partition=kate_reserved
#SBATCH --job-name=train_hiveformer
#SBATCH --output=slurm_logs/train-hiveformer-%j.out
#SBATCH --error=slurm_logs/train-hiveformer-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=62gb
#SBATCH --mem-per-gpu=32gb
#SBATCH --exclude matrix-0-24

python train.py "$@"
