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

if [ -f $tasks ]; then
tasks=$(sed -n "${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p" $tasks)
fi

seed=${seed:-0}
seed_gen=${seed_gen:-$seed}
variations=${variations:-0}
log_dir=$HOME/src/vln-robot/alhistory/logs
data_dir=$SCRATCH/datasets/vln-robot/c2farm/$seed
output_dir=${output_dir:-$HOME/src/vln-robot/alhistory/dataset-$seed-${seed_gen}}
mkdir -p $data_dir
mkdir -p $output_dir
module load singularity


pushd $HOME/src/vln-robot/alhistory/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	    /usr/bin/python3.9 data_gen.py \
		--data_dir=$data_dir \
		--output=$output_dir \
		--tasks=$tasks \
		--seed=$seed_gen
popd
