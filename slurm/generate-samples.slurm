#!/bin/bash
#SBATCH --job-name=generate
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

hf_dir=$HOME/src/hiveformer
rlbench_dir=$hf_dir/RLBench
log_dir=$SCRATCH/logs/hiveformer
data_dir=$SCRATCH/datasets/hiveformer/raw/seed$seed/
store_dir=$STORE/datasets/hiveformer/raw/seed$seed/
output_dir=${output_dir:-$SCRATCH/datasets/pkg/seed-${seed_gen}}
tar_file=$output_dir/${task}+${variation}.tar.gz

mkdir -p $data_dir
mkdir -p $log_dir
mkdir -p $store_dir
mkdir -p $output_dir

module load singularity

# Clean up current mess
if [ -d $data_dir/$task/variation$variation ]; then
  rm -r $data_dir/$task/variation$variation
fi

# Quit if the task is broken
is_broken=$(python -c "import json; ep=json.load(open('$alh_dir/episodes.json')); print('$task' in ep['broken']);")
if [ $is_broken = True ]; then
	echo "Broken task" $task
	exit 1
fi

# Quit if the task is already stored
if [ ! -z ${quit_if_exists+x} ] && [ "$quit_if_exists" = true ] && [ -f "$tar_file" ];then
	echo "The tar_file folder already exists"
	exit 0
fi

# Uncompress an existing stored file
if [ -f $tar_file ]; then
  echo "Untar $task/$variation"
  tar -xzf $tar_file -C $data_dir
fi

# Generate samples
pushd $rlbench_dir/tools/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/hiveformer.sif \
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

# Package samples
pushd $hf_dir
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/hiveformer.sif \
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

# Store samples
pushd $data_dir 
  if [ -f $tar_file ]; then
    echo "Remove old tar $tar_file"
    rm $tar_file
  fi
  tar -czf $tar_file $task/variation$variation
  rm -r $data_dir/$task/variation$variation
popd
