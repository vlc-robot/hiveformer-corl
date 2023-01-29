#!/bin/bash
#SBATCH --job-name=dataset
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

checkpoints=${checkpoints:-checkpoints.txt}
checkpoint_task=$(sed -n "${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p" $checkpoints)
checkpoint=$(echo $checkpoint_task | cut -d ',' -f 1)
task=$(echo $checkpoint_task | cut -d ',' -f 2)
name=$(echo $checkpoint | cut -d '/' -f 2)
seed=${seed:-0}
log_dir=$HOME/src/vln-robot/alhistory/logs
 
 
pwd; hostname; date
cd $HOME/src/vln-robot/alhistory/

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	/usr/bin/python3.9 $HOME/src/vln-robot/alhistory/eval.py \
		--seed $seed \
		--checkpoint $HOME/src/vln-robot/alhistory/$checkpoint \
		--tasks $task \
		--name ${name}-replay \
		--headless \
		--record_actions \
 		--replay_actions $HOME/src/vln-robot/alhistory/xp/$name-record/test-version0/$task
