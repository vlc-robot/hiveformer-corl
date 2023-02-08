#!/bin/sh

main_dir=02_08

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir $task-WITH-gt_ghost-NO-nonsupervisedball \
     --use_ground_truth_position_for_sampling 1 \
     --non_supervised_ball_radius 0
done

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir $task-NO-gt_ghost-NO-nonsupervisedball \
     --use_ground_truth_position_for_sampling 0 \
     --non_supervised_ball_radius 0
done
