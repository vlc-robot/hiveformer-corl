#!/bin/sh

main_dir=02_04_more

# Cross-entropy

#task_file=debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --exp_log_dir $main_dir/xentropy_non_supervised_ball_001 \
#     --run_log_dir $task \
#     --non_supervised_ball_radius 0.01
#done
#
#task_file=debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --exp_log_dir $main_dir/xentropy_non_supervised_ball_003 \
#     --run_log_dir $task \
#     --non_supervised_ball_radius 0.03
#done

# BCE

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir/bce_with_gt \
     --run_log_dir $task \
     --non_supervised_ball_radius 0.01 \
     --position_loss bce \
     --use_ground_truth_position_for_sampling 1
done

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir/bce_no_gt \
     --run_log_dir $task \
     --non_supervised_ball_radius 0.01 \
     --position_loss bce \
     --use_ground_truth_position_for_sampling 0
done

# CE

task_file=more_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir/ce_with_gt \
     --run_log_dir $task \
     --non_supervised_ball_radius 0.01 \
     --position_loss ce \
     --use_ground_truth_position_for_sampling 1
done

task_file=more_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir/ce_no_gt \
     --run_log_dir $task \
     --non_supervised_ball_radius 0.01 \
     --position_loss ce \
     --use_ground_truth_position_for_sampling 0
done
