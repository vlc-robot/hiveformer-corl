#!/bin/bash
#SBATCH --job-name=vlm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH -p gpu_p13
#SBATCH --time 20:00:00

set -x
set -e

module purge
module load singularity
export PYTHONPATH=/opt/YARR/
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
pwd; hostname; date

alhistory=$HOME/src/vln-robot/alhistory
xp=${xp:-$alhistory/xp/}
log_dir=$alhistory/logs
seed=${seed:-2}

if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  if [ -f $tasks ]; then
  	task_file=$tasks
  	num_tasks=$(wc -l < $task_file)
  	task_id=$(( (SLURM_ARRAY_TASK_ID % num_tasks) +1 ))
  	tasks=$(sed -n "${task_id},${task_id}p" $tasks)
  else
  	num_tasks="${#tasks[@]}"
  fi
  seed=$(( SLURM_ARRAY_TASK_ID  / num_tasks ))
fi

dataset=${dataset:-$alhistory/datasets/dataset-$seed}
name=${name:-${prefix}${tasks}}

if [ ! -z $num_seeds ]; then
	valset=${valset:-$alhistory/datasets/dataset-$((($seed + 1) % $num_seeds))}
fi

mkdir -p $log_dir
mkdir -p $xp

rm /tmp/.X99-lock || true

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	/usr/bin/python3.9 $alhistory/multi_task_baselines.py \
		--name $name \
		--seed $seed \
		--tasks $tasks \
		--dataset $dataset \
		--valset $valset \
		--xp $xp \
		--headless \
		$args
