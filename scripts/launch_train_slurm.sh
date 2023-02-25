#!/bin/sh

task_file=tasks/10_autolambda_tasks.csv

main_dir=02_25_increase_visual_resolution

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
for task in take_money_out_safe put_knife_on_chopping_board; do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 16 \
     --num_workers 4 \
     --position_prediction_only 1 \
     --run_log_dir $task-$image_size
done

dataset=/home/tgervet/datasets/hiveformer/packaged/0
valset=/home/tgervet/datasets/hiveformer/packaged/1
image_size="128,128"
for task in take_money_out_safe put_knife_on_chopping_board; do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 16 \
     --num_workers 4 \
     --position_prediction_only 1 \
     --run_log_dir $task-$image_size
done
