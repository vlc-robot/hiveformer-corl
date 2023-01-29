#!/bin/bash
#SBATCH --job-name=c2farm
#SBATCH --ntasks=1          
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH -A vuo@cpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
# #SBATCH -C v100-32g
#SBATCH -p gpu_p13
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
 
# go into the submission directory
set -x
set -e

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=/opt/YARR/
num_episodes=${num_episodes:-10}
variation=${variation:-0}
echo $task
if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  num_tasks=$(wc -l < $task_file)
  task_id=$(( (SLURM_ARRAY_TASK_ID % num_tasks) +1 ))
  task=$(sed -n "${task_id},${task_id}p" $task_file)
  seed=$(( (SLURM_ARRAY_TASK_ID - 1) / num_tasks ))
else
  seed=${seed:-0}
  log_dir=$alhistory/logs
fi

JOB_LOG_DIR=${JOB_LOG_DIR:-$HOME/src/vln-robot/alhistory/logs}
data_dir=$SCRATCH/datasets/vln-robot/c2farm/100/seed${seed}
log_dir=$SCRATCH/logs/vln-robot/c2farm/$num_episodes/seed${seed}
store_dir=$STORE/datasets/vln-robot/c2farm/100/seed$seed/
method=${method:-C2FARM}
mkdir -p $data_dir $log_dir $data_dir $JOB_LOG_DIR

if grep -Fxq "$task" multiviews.csv; then
	cameras="wrist left_shoulder right_shoulder"
else
	cameras=front
fi
echo ${cameras[@]}

# Clean up mess
if [ -d $data_dir/$task ]; 
	rm -r $data_dir/$task
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
  tar -xzf $tar_file -C $data_dir
fi

module load singularity

pushd $HOME/src/vln-robot/ARM2/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a -e $JOB_LOG_DIR/$SLURM_JOBID.err \
	    /usr/bin/python3.9 launch.py \
		method=$method \
		rlbench.task=$task \
		rlbench.demo_path=$data_dir \
		rlbench.demos=$num_episodes \
		framework.gpu=0 \
		framework.logdir=$log_dir
	# 	rlbench.cameras="$cameras" \
popd
