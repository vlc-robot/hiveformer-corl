#!/bin/sh

task_file=tasks/10_autolambda_tasks.csv

#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --image_size "128,128" \
#     --exp_log_dir $main_dir \
#     --model original \
#     --run_log_dir HIVEFORMER-$task
#done

main_dir=02_24_improve_position_baseline
for fine_sampling_ball_diameter in 0.08 0.10; do
  for task in take_money_out_safe put_knife_on_chopping_board; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
       --valset /home/tgervet/datasets/hiveformer/packaged/3 \
       --image_size "256,256" \
       --exp_log_dir $main_dir \
       --model baseline \
       --batch_size 15 \
       --num_workers 2 \
       --position_prediction_only 1 \
       --fine_sampling_ball_diameter $fine_sampling_ball_diameter \
       --run_log_dir $task-$fine_sampling_ball_diameter
  done
done

#main_dir=02_24_analogical_poc_for_all_tasks
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_4gpu_12gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/3 \
#     --image_size "256,256" \
#     --exp_log_dir $main_dir \
#     --model analogical \
#     --batch_size 20 \
#     --num_workers 8 \
#     --position_prediction_only 1 \
#     --rotation_parametrization quat_from_top_ghost \
#     --support_set rest_of_batch \
#     --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#     --run_log_dir $task
#done
