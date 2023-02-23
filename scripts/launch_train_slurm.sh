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

main_dir=02_20_compare_hiveformer_and_baseline
#for task in $(cat $task_file | tr '\n' ' '); do
for task in reach_target slide_block_to_target take_money_out_safe take_umbrella_out_of_umbrella_stand; do
  sbatch train_1gpu_32gb_kate.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
     --valset /home/tgervet/datasets/hiveformer/packaged/3 \
     --image_size "256,256" \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 15 \
     --num_workers 2 \
     --position_prediction_only 0 \
     --run_log_dir BASELINE-WITH-100-ROTATION-SCALING-$task
done

#main_dir=02_23_analogical_poc_for_all_tasks
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/3 \
#     --image_size "256,256" \
#     --exp_log_dir $main_dir \
#     --model analogical \
#     --batch_size 15 \
#     --num_workers 2 \
#     --position_prediction_only 1 \
#     --rotation_parametrization quat_from_top_ghost \
#     --support_set rest_of_batch \
#     --run_log_dir $task
#done
