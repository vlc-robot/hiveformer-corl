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
 

# This script generates fresh samples
# OOM errors with CPU

# go into the submission directory
set -x
set -e

  

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=/opt/YARR/
num_episodes=${num_episodes:-10}

if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  task_file=$tasks
  num_tasks=$(wc -l < $task_file)
  task_id=$(( (${SLURM_ARRAY_TASK_ID} % $num_tasks) +1 ))
  taskvar=$(sed -n "${task_id},${task_id}p" $tasks)
  task=$(echo $taskvar | awk -F ',' '{ print $1 }')
  variation=$(echo $taskvar | awk -F ',' '{ print $2 }')
  variation=${variation:-0}
  seed_default=$(( ${SLURM_ARRAY_TASK_ID} / $num_tasks ))
  seed=${seed:-$seed_default}
  seed_gen=${seed:-$seed_gen}
else
  seed=${seed:-0}
  seed_gen=${seed:-$seed_gen}
  variation=${variation:-0}
  offset=${seed:-$seed_gen}
fi

alh_dir=$HOME/src/vln-robot/alhistory
log_dir=$HOME/src/vln-robot/alhistory/logs
data_dir=$SCRATCH/datasets/vln-robot/c2farm2/$num_episodes/seed$seed/
store_dir=$STORE/datasets/vln-robot/c2farm2/$num_episodes/seed$seed/
output_dir=${output_dir:-$SCRATCH/datasets/vln-robot/alhistory/dataset2-${seed_gen}}

mkdir -p $data_dir
mkdir -p $log_dir
mkdir -p $store_dir
mkdir -p $output_dir

module load singularity

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
if [ ! -z ${quit_if_exists+x} ] && [ "$quit_if_exists" = true ] && [ -f "$dest" ];then
	echo "The dest folder already exists"
	exit 0
fi

if [ -f $store_dir/$task-$variation.tar.gz ]; then
  echo "Untar $task/$variation"
  tar -xzf $store_dir/$task-$variation.tar.gz -C $data_dir
fi

pushd $HOME/src/vln-robot/RLBench/tools/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	    /usr/bin/python3.9 dataset_generator.py \
		--save_path=$data_dir \
		--tasks=$task \
		--image_size=128,128 \
		--renderer=opengl \
		--episodes_per_task=$(($num_episodes * 12 / 10))  \
		--variations=$(( $variation + 1 )) \
		--offset=$variation \
		--processes=1
popd


pushd $HOME/src/vln-robot/alhistory/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	    /usr/bin/python3.9 data_gen.py \
		--data_dir=$data_dir \
		--output=$output_dir \
		--max_variations=$(( $variation + 1 )) \
		--offset=$variation \
		--num_episodes=$num_episodes \
		--tasks=$task \
		--seed=$seed \
		$pack_args
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
