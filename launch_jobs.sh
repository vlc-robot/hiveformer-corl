#!/bin/sh

main_dir=02_08_debugged

task_file=debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
for task in put_money_in_safe; do
  for rad in 0 0.01; do
    for sample in 0 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
         --valset /home/tgervet/datasets/hiveformer/packaged/1 \
         --exp_log_dir $main_dir \
         --run_log_dir $task-rad-$rad-sample-$sample \
         --ground_truth_ball_radius $rad \
         --use_ground_truth_position_for_sampling $sample
    done
  done
done
