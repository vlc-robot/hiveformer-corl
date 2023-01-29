#!/bin/bash
#SBATCH --job-name=c2farm
#SBATCH --ntasks=1          
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH -A vuo@cpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -p gpu_p13
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
 

# This script packages sim2real samples

set -x
set -e

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=$PYTHONPATH:$HOME/src/muse

# offset=$((SLURM_ARRAY_TASK_ID * 10))
# num_episodes=$((offset + 10))

module load singularity
alhistory=$HOME/src/vln-robot/alhistory/
log_dir=$alhistory/logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genini01/uok79zh/.mujoco/mujoco210/bin:/usr/lib/nvidia

pushd $alhistory
/gpfswork/rech/vuo/uok79zh/anaconda3/envs/h/bin/python3.9 \
	data_gen_sim2real.py \
		--data_dir=$data_dir \
		--tasks="$tasks" \
		--output=$output \
		# --offset_episode=$offset \
		# --num_episodes=$num_episodes
popd


pushd $data_dir 
  tar_file=$store_dir/$task-$variation.tar.gz
  if [ -f $tar_file ]; then
    echo "Remove old tar $tar_file"
    rm $tar_file
  fi
  tar -czf $tar_file $task/variation$variation
  rm -r $data_dir/$task/variation$variation
popd
