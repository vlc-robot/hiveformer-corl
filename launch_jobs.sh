#!/bin/sh

main_dir=02_04_no_ghost_points

task_file=10_autolambda_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir $task \
     --position_loss mse \
     --sample_ghost_points 0
done
