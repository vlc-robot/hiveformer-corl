#!/bin/bash
#SBATCH --job-name=s2r-gen
#SBATCH --ntasks=1          
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
 

# This script generates and packages samples from MuJoCo

set -x
set -e

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
num_episodes=${num_episodes:-500}

if [ -z $SLURM_ARRAY_TASK_ID ]; then
	echo 'Please provide a task job file'
	exit 1;
fi

task_file=$tasks
num_tasks=$(wc -l < $task_file)
task_id=$(( (${SLURM_ARRAY_TASK_ID} % $num_tasks) +1 ))

envtaskvar=$(sed -n "${task_id},${task_id}p" $tasks)
env_name=$(echo $envtaskvar | awk -F ',' '{ print $1 }')
task=$(echo $envtaskvar | awk -F ',' '{ print $2 }')
variation=$(echo $envtaskvar | awk -F ',' '{ print $3 }')

seed_default=$(( ${SLURM_ARRAY_TASK_ID} / $num_tasks ))
seed=${seed:-$seed_default}

alh_dir=$HOME/src/vln-robot/alhistory
log_dir=$HOME/src/vln-robot/alhistory/logs
data_dir=${data_dir:-$SCRATCH/datasets/vln-robot/sim2real/raw/seed$seed/}
output_dir=${output_dir:-$SCRATCH/datasets/vln-robot/alhistory/sim2real/datasets-dr/dataset-${seed}}

mkdir -p $data_dir
mkdir -p $log_dir
mkdir -p $output_dir

# Clean up current mess
if [ -d $data_dir/$task/variation$variation ]; then
  rm -r $data_dir/$task/variation$variation
fi

is_broken=$(python -c "import json; ep=json.load(open('$alh_dir/episodes.json')); print('$task' in ep['broken']);")
if [ $is_broken = True ]; then
	echo "Broken task" $task
	exit 1
fi

dest=$output_dir/${task}+${variation}
if [ ! -z ${quit_if_exists+x} ] && [ "$quit_if_exists" = true ] && [ -d "$dest" ];then
	echo "The dest folder already exists"
	exit 0
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

$WORK/anaconda3/envs/h/bin/python3.9 \
  	collect_sim2real.py \
  	--output $data_dir/$task \
  	--num_episodes $num_episodes \
  	--seed $seed \
  	--env_name $env_name \
  	--num_workers 4 \
	--offset $variation \
  	--num_variations 1

pushd $HOME/src/vln-robot/alhistory/
export PYTHONPATH=$PYTHONPATH:$HOME/src/muse
$WORK/anaconda3/envs/h/bin/python3.9 \
	data_gen_sim2real.py \
		--data_dir=$data_dir \
		--output=$output_dir \
		--variations=$variation \
		--num_workers 4 \
		--tasks=$task \
		--seed=$seed \
		$pack_args
popd

pushd $data_dir 
  rm -r $data_dir/$task/variation$variation
popd
