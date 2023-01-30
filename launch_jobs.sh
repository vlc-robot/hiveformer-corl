#!/bin/sh

root=/home/tgervet
train_seed=0
val_seed=1
task_file=10_autolambda_tasks.csv
output_dir=$root/datasets/hiveformer/packaged

#experiment=reproduce_uncleaned_with_released_exp1
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu.sh \
#    --name $experiment \
#    --tasks $task \
#    --dataset $output_dir/$train_seed \
#    --valset $output_dir/$val_seed
#done

experiment=ghost_points_exp2
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
    --name $experiment \
    --tasks $task \
    --dataset $output_dir/$train_seed \
    --valset $output_dir/$val_seed \
    --model develop \
    --batch_size 10 \
    --lr 0.001
done
