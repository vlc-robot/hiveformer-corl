#!/bin/sh

root=/home/tgervet
train_seed=0
val_seed=1
task_file=10_autolambda_tasks.csv
output_dir=$root/datasets/hiveformer/packaged

#experiment=exp5
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu.sh \
#    --name $experiment \
#    --tasks $task \
#    --dataset $output_dir/$train_seed \
#    --valset $output_dir/$val_seed
#done

experiment=exp8
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu.sh \
    --name $experiment \
    --tasks $task \
    --dataset $output_dir/$train_seed \
    --valset $output_dir/$val_seed \
    --model develop
done
