#!/bin/bash
#SBATCH --job-name=vlm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
# #SBATCH -C v100-32g
#SBATCH -p gpu_p13
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint nomultithread
#SBATCH --time 20:00:00
#SBATCH --exclusive
#SBATCH --qos=qos_gpu-t3


set -x

export WORK=/gpfswork/rech/vuo/uok79zh/
export HOME=/linkhome/rech/genini01/uok79zh/
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1

module load git

name=${name:-''}

srun --export=ALL $WORK/anaconda3/envs/h/bin/python train.py \
	--gpus 4 \
	--num-nodes 1 \
	--train True \
	--name $name \
	$args
