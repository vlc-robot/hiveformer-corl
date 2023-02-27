#!/bin/sh

main_dir=02_26_MATCH_HIVEFORMER
task_file=tasks/7_interesting_tasks.csv
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --exp_log_dir $main_dir \
     --run_log_dir $task
done
