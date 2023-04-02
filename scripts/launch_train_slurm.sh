#!/bin/sh

# --------------------------------------------------------------------------------------------
# Training configurations
# --------------------------------------------------------------------------------------------
# Single task
#num_workers=2
#batch_size=3
#accumulate_grad_batches=2
#task_file=tasks/7_interesting_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --num_workers $num_workers \
#   --batch_size $batch_size \
#   --accumulate_grad_batches $accumulate_grad_batches \
#   --run_log_dir $task
#done

# Multi task
#embedding_dim=120
#num_workers=12
#batch_size=16
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#sbatch train_4gpu_12gb.sh \
#  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#  --tasks $(cat $task_file | tr '\n' ' ') \
#  --dataset $dataset \
#  --valset $valset \
#  --exp_log_dir $main_dir \
#  --num_workers $num_workers \
#  --batch_size $batch_size \
#  --train_iters $train_iters \
#  --embedding_dim $embedding_dim \
#  --run_log_dir BASELINE-MULTI-TASK

# Analogy
#embedding_dim=120
#num_workers=12
#batch_size=8
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#sbatch train_4gpu_12gb.sh \
#  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#  --model analogical \
#  --rotation_parametrization "quat_from_top_ghost" \
#  --support_set "others" \
#  --tasks $(cat $task_file | tr '\n' ' ') \
#  --dataset $dataset \
#  --valset $valset \
#  --exp_log_dir $main_dir \
#  --num_workers $num_workers \
#  --batch_size $batch_size \
#  --train_iters $train_iters \
#  --embedding_dim $embedding_dim \
#  --run_log_dir ANALOGY-MULTI-TASK

# PerAct multi-variation
#--gripper_loc_bounds_file tasks/18_peract_tasks_location_bounds.json
#--variations {0..199}
# --------------------------------------------------------------------------------------------

#main_dir=03_07_hiveformer_10_demos
#task_file=tasks/7_interesting_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --image_size "128,128" \
#     --exp_log_dir $main_dir \
#     --model original \
#     --max_episodes_per_taskvar 10 \
#     --run_log_dir $task
#done

#main_dir=03_24_eval_on_peract_18_tasks
#num_workers=1
#use_instruction=1
#train_iters=500_000
#task_file=tasks/14_peract_short_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_val
##for task in $(cat $task_file | tr '\n' ' '); do
#for task in light_bulb_in put_groceries_in_cupboard place_shape_in_shape_sorter; do
#  sbatch train_1gpu_12gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --num_workers $num_workers \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size \
#   --train_iters $train_iters \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --variations {0..199} \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --run_log_dir $task
#done

# TODO
#main_dir=03_24_hiveformer_setting
#use_instruction=0
#task_file=tasks/hiveformer_74_tasks_61_74.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --run_log_dir $task-HIVEFORMER
#done

main_dir=04_02_multi_task
use_instruction=1
embedding_dim=120
num_workers=16
train_iters=2_000_000
task_file=tasks/autolambda_10_tasks.csv
gripper_loc_bounds_file=tasks/10_autolambda_tasks_location_bounds.json
dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val

batch_size=16
batch_size_val=4
model=baseline
sbatch train_4gpu_32gb.sh \
 --model $model \
 --devices cuda:0 cuda:1 cuda:2 cuda:3 \
 --tasks $(cat $task_file | tr '\n' ' ') \
 --dataset $dataset \
 --valset $valset \
 --exp_log_dir $main_dir \
 --num_workers $num_workers \
 --use_instruction $use_instruction \
 --embedding_dim $embedding_dim \
 --train_iters $train_iters \
 --gripper_loc_bounds_file $gripper_loc_bounds_file \
 --use_instruction $use_instruction \
 --logger wandb \
 --run_log_dir multitask-$model

batch_size=8
batch_size_val=2
model=analogical
sbatch train_4gpu_32gb.sh \
 --model $model \
 --devices cuda:0 cuda:1 cuda:2 cuda:3 \
 --tasks $(cat $task_file | tr '\n' ' ') \
 --dataset $dataset \
 --valset $valset \
 --exp_log_dir $main_dir \
 --num_workers $num_workers \
 --use_instruction $use_instruction \
 --embedding_dim $embedding_dim \
 --train_iters $train_iters \
 --gripper_loc_bounds_file $gripper_loc_bounds_file \
 --use_instruction $use_instruction \
 --logger wandb \
 --run_log_dir multitask-$model

# TODO
#main_dir=03_30_hiveformer_hard_10_demo_tasks
#use_instruction=0
#task_file=tasks/hiveformer_hard_10_demo_tasks.csv
#gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
#for task in $(cat $task_file | tr '\n' ' '); do
#  for point_cloud_rotate_yaw_range in 0.0 45.0; do
#    sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset $dataset \
#     --valset $valset \
#     --exp_log_dir $main_dir \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file \
#     --use_instruction $use_instruction \
#     --logger wandb \
#     --run_log_dir $task-10-demo-$point_cloud_rotate_yaw_range \
#     --max_episodes_per_task 10 \
#     --point_cloud_rotate_yaw_range $point_cloud_rotate_yaw_range
#  done
#done
