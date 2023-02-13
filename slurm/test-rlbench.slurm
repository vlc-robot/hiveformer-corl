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
seed=${seed:-0}
seed_gen=${seed:-$seed_gen}
num_repeats=${num_repeats:-5}
num_variations=${num_variations:-1}
log_dir=$HOME/src/vln-robot/alhistory/logs
data_dir=$SCRATCH/datasets/vln-robot/c2farm/$num_episodes/seed$seed/
output_dir=${output_dir:-$HOME/src/vln-robot/alhistory/dataset-${seed_gen}}
store_dir=$STORE/datasets/vln-robot/c2farm/$num_episodes/seed$seed/
if [ -f $tasks ]; then
	tasks=($(sed -n "${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p" $tasks))
fi
mkdir -p $data_dir
mkdir -p $output_dir
mkdir -p $store_dir
module load singularity

for task in "${tasks[@]}"; do
  if [ ! -d $data_dir/$task ]; then
    echo "Untar $task"
    tar -xzf $store_dir/$task.tar.gz -C $data_dir
  fi
done

for i in $(seq -s ' ' 1 $num_repeats); do
echo "TRIAL NUMBER $i"
cd $HOME/src/vln-robot/alhistory
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE \
	$SINGULARITY_ALLOWED_DIR/vln-robot.sif \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	    /usr/bin/python3.9 $HOME/src/vln-robot/alhistory/test-rlbench.py \
		--data_dir=$data_dir \
		--tasks=$tasks \
		--seed=$seed \
		--headless \
		$args
done

cd $data_dir
for task in "${tasks[@]}"; do
  if [ ! -f $store_dir/$task.tar.gz ]; then
    echo "Tar $task"
    tar -czf $store_dir/$task.tar.gz $task
  fi
  rm -r $data_dir/$task
done
