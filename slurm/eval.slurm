#!/bin/bash
#SBATCH --job-name=test-alhistory
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -C v100-32g
#SBATCH -c 10
#SBATCH -A vuo@v100
#SBATCH --hint nomultithread
#SBATCH --time 20:00:00
#SBATCH --qos=qos_gpu-t3

set -x
module purge
module load singularity
export PYTHONPATH=/opt/YARR/
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
pwd; hostname; date
seed=${seed:-2}
log_dir=$HOME/src/vln-robot/alhistory/logs
if [ -f $tasks ]; then
tasks=$(sed -n "${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p" $tasks)
fi
checkpoint=${checkpoint:-${prefix}${tasks}}
num_episodes=${num_episodes:-100}

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	/usr/bin/python3.9 $HOME/src/vln-robot/alhistory/eval.py \
		--seed $seed \
		--checkpoint $checkpoint \
		--tasks $tasks \
		--num_episodes $num_episodes \
		--headless
