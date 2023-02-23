#!/bin/bash

#SBATCH --partition=russ_reserved
#SBATCH --partition=kate_reserved
#SBATCH --job-name=train_hiveformer
#SBATCH --output=slurm_logs/train-hiveformer-%j.out
#SBATCH --error=slurm_logs/train-hiveformer-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --nodelist matrix-1-14
#SBATCH --nodelist matrix-1-24
#SBATCH --nodelist matrix-2-25
#SBATCH --nodelist matrix-2-29
#SBATCH --nodelist matrix-1-16

python train.py "$@"
