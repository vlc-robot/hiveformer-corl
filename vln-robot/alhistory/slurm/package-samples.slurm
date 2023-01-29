#!/bin/bash
#SBATCH --job-name=c2farm
#SBATCH --ntasks=1          
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH -A vuo@cpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -C v100-32g
#SBATCH -p gpu_p13
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
 
# This script packages samples that were already generated.
# OMM error with cpu partition...

# go into the submission directory
set -x
set -e

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=/opt/YARR/
num_workers=${num_workers:-10}
num_episodes=100

# SLURM_ARRAY_TASK_ID starts at 1
if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  task_file=$tasks
  num_taskvars=$(wc -l < $task_file)
  task_id=$(( (${SLURM_ARRAY_TASK_ID} % $num_taskvars) +1 ))
  taskvar=$(sed -n "${task_id},${task_id}p" $tasks)
  task=$(echo $taskvar | awk -F ',' '{ print $1 }')
  variation=$(echo $taskvar | awk -F ',' '{ print $2 }')
  variation=${variation:-0}
  seed_default=$(( ( ${SLURM_ARRAY_TASK_ID} - 1 ) / $num_taskvars ))
  seed=${seed:-$seed_default}
else
  seed=${seed:-0}
  variation=${variation:-0}
fi

log_dir=$HOME/src/vln-robot/alhistory/logs
data_dir=$SCRATCH/datasets/vln-robot/c2farm/$num_episodes/seed$seed
store_dir=$STORE/datasets/vln-robot/c2farm/$num_episodes/seed$seed
output_dir=${output_dir:-$SCRATCH/datasets/vln-robot/alhistory/dataset-${seed}}
var_dir=$data_dir/$task/variation$variation

mkdir -p $data_dir
mkdir -p $log_dir
mkdir -p $store_dir
mkdir -p $output_dir

dest=$output_dir/${task}+${variation}
if [ ! -z ${quit_if_exists+x} ] && [ "$quit_if_exists" = true ] && [ -f "$dest" ];then
	echo "The dest folder already exists"
	exit 0
fi

module load singularity

echo "Starting from a clean state"
if [ -d $var_dir ]; then
  rm -r $var_dir
fi

echo "Untar $task/variation$variation"
tar_file=$store_dir/$task-$variation.tar.gz
if [ ! -f $tar_file ]; then
  tar_file=$store_dir/$task-$variation.tar.gz
fi
if [ ! -f $tar_file ]; then
  echo 'No tar file exists!'
  exit 1
else
  mkdir -p $var_dir 
  tar xf $tar_file -C $var_dir --strip-components 2 
fi


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
		--num_workers=$num_workers \
		--tasks=$task \
		--seed=$seed \
		$pack_args
popd

pushd $data_dir 
  tar_file=$store_dir/$task-$variation.tar.gz
  if [ ! -f $tar_file ]; then
    echo "Tar $task"
    tar -czf $tar_file $task
  fi

  rm -r $var_dir
popd
