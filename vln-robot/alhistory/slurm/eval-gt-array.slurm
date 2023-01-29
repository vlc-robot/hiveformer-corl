#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --ntasks=1          
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -p gpu_p13
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
 
module purge
module load singularity
export PYTHONPATH=/opt/YARR/
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

set -x
set -e 

task=$(sed -n "${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p" $task_file)
name=${name:-eval-gt-$task}
variation=${variation:-0}
seed=0
 
hostname; date
cd $HOME/src/vln-robot/alhistory/

alh_dir=$HOME/src/vln-robot/alhistory
log_dir=$alh_dir/logs
data_dir=$alh_dir/demos
store_dir=$STORE/datasets/vln-robot/c2farm/100/seed$seed

if [ ! -f $store_dir/$task-$variation.tar.gz ]; then
  echo "Can't find data $store_dir/$task-$variation.tar.gz"
  exit 1
fi
echo "Untar $task/$variation"
tar -xzf $store_dir/$task-$variation.tar.gz -C $data_dir

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	/usr/bin/python3.9 $HOME/src/vln-robot/alhistory/eval.py \
		--seed $seed \
		--checkpoint $checkpoint \
		--name $prefix$name \
		--tasks $task \
		--num_episodes 100 \
		--headless \
		--ground_truth_position \
		--ground_truth_rotation \
		--ground_truth_gripper \


rm -r $data_dir/$task/variation$variation
