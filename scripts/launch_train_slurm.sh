#!/bin/sh

main_dir=02_26_match_hiveformer
task_file=tasks/7_interesting_tasks.csv
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 16 \
     --num_workers 2 \
     --position_prediction_only 0 \
     --rotation_loss_coeff 10 \
     --fine_sampling_ball_diameter 0.16 \
     --run_log_dir FULL-$task-BALL-$fine_sampling_ball_diameter
done

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_4gpu_12gb.sh \
     --devices cuda:0 cuda:1 cuda:2 cuda:3 \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 16 \
     --num_workers 2 \
     --position_prediction_only 0 \
     --rotation_loss_coeff 10 \
     --fine_sampling_ball_diameter 0.08 \
     --run_log_dir FULL-$task-BALL-$fine_sampling_ball_diameter
done

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_4gpu_12gb.sh \
     --devices cuda:0 cuda:1 cuda:2 cuda:3 \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 16 \
     --num_workers 2 \
     --position_prediction_only 1 \
     --rotation_loss_coeff 10 \
     --fine_sampling_ball_diameter 0.16 \
     --run_log_dir POSITION-ONLY-$task-BALL-$fine_sampling_ball_diameter
done
