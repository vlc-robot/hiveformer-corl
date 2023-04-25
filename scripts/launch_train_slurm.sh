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

# PERACT
#main_dir=03_24_eval_on_peract_18_tasks
#use_instruction=1
#task_file=tasks/peract_18_tasks_1_10.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_val
#train_iters=200_000
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --variations {0..199} \
#   --train_iters $train_iters \
#   --run_log_dir $task-PERACT
#done

# HIVEFORMER
main_dir=03_24_hiveformer_setting
use_instruction=0
task_file=tasks/hiveformer_74_tasks_61_74.csv
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
train_iters=400_000
#for task in place_shape_in_shape_sorter sweep_to_dustpan take_plate_off_colored_dish_rack place_hanger_on_rack; do
#for task in plug_charger_in_power_supply reach_and_drag reach_and_drag setup_checkers tower3 straighten_rope; do
for task in take_shoes_out_of_box; do # stack_cups stack_blocks tv_on; do  # TODO Likely OOM error, try cache_size_val=0
#for task in stack_cups stack_blocks wipe_desk tv_on; do
  sbatch train_1gpu_32gb_125gb.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --use_instruction $use_instruction \
   --logger wandb \
   --train_iters $train_iters \
   --run_log_dir $task-HIVEFORMER \
   --cache_size_val 0
done

# Multi-task
#main_dir=04_03_multi_task
#use_instruction=1
#embedding_dim=120
#cache_size=0
#cache_size_val=0
#train_iters=2_000_000
#task_file=tasks/autolambda_10_tasks.csv
#gripper_loc_bounds_file=tasks/10_autolambda_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/74_hiveformer_tasks_val
#
#batch_size=8
#batch_size_val=2
#model=baseline
#num_sampling_level=3
#regress_position_offset=0
#for num_workers in 1 2 4 8; do
#  sbatch train_1gpu_32gb.sh \
#   --model $model \
#   --tasks $(cat $task_file | tr '\n' ' ') \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --use_instruction $use_instruction \
#   --embedding_dim $embedding_dim \
#   --train_iters $train_iters \
#   --cache_size $cache_size \
#   --cache_size_val $cache_size_val \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size_val \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --use_instruction $use_instruction \
#   --logger wandb \
#   --num_sampling_level $num_sampling_level \
#   --num_workers $num_workers \
#   --regress_position_offset $regress_position_offset \
#   --run_log_dir multitask-$model-$num_workers-workers
#done

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
