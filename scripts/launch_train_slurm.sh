#!/bin/sh

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
main_dir=exp

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

#main_dir=03_06_task_specific_biases2
#embedding_dim=120
#num_workers=12
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#
#batch_size=8
#for task_specific_biases in 0 1; do
#  sbatch train_4gpu_12gb.sh \
#    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#    --model analogical \
#    --rotation_parametrization "quat_from_top_ghost" \
#    --support_set "others" \
#    --tasks $(cat $task_file | tr '\n' ' ') \
#    --dataset $dataset \
#    --valset $valset \
#    --exp_log_dir $main_dir \
#    --num_workers $num_workers \
#    --batch_size $batch_size \
#    --train_iters $train_iters \
#    --embedding_dim $embedding_dim \
#    --run_log_dir ANALOGY-task_specific_biases-$task_specific_biases
#done
#
#batch_size=16
#for task_specific_biases in 0 1; do
#  sbatch train_4gpu_12gb.sh \
#    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#    --model baseline \
#    --tasks $(cat $task_file | tr '\n' ' ') \
#    --dataset $dataset \
#    --valset $valset \
#    --exp_log_dir $main_dir \
#    --num_workers $num_workers \
#    --batch_size $batch_size \
#    --train_iters $train_iters \
#    --embedding_dim $embedding_dim \
#    --run_log_dir BASELINE-task_specific_biases-$task_specific_biases
#done

#main_dir=03_07_hiveformer_10_demos_new
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

#main_dir=03_08_data_augmentations3
#num_workers=2
#batch_size=3
#accumulate_grad_batches=2
#num_sampling_level=2
#regress_position_offset=1
#for task in pick_and_lift pick_up_cup; do
#  for image_rescale in "1.0,1.0"; do
#    for point_cloud_rotate_yaw_range in 0.0 45.0; do
#      sbatch train_1gpu_12gb.sh \
#       --tasks $task \
#       --dataset $dataset \
#       --valset $valset \
#       --exp_log_dir $main_dir \
#       --num_workers $num_workers \
#       --batch_size $batch_size \
#       --accumulate_grad_batches $accumulate_grad_batches \
#       --num_sampling_level $num_sampling_level \
#       --regress_position_offset $regress_position_offset \
#       --image_rescale $image_rescale \
#       --point_cloud_rotate_yaw_range $point_cloud_rotate_yaw_range \
#       --run_log_dir $task-rescale-$image_rescale-rotate-$point_cloud_rotate_yaw_range
#    done
#  done
#done

#main_dir=03_09_peract_setting
#num_workers=2
#batch_size=3
#accumulate_grad_batches=2
#num_sampling_level=2
#regress_position_offset=1
#use_instruction=1
#task_file=tasks/18_peract_tasks.csv
#gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
#dataset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_train
#valset=/projects/katefgroup/analogical_manipulation/rlbench/packaged/18_peract_tasks_val
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --num_workers $num_workers \
#   --batch_size $batch_size \
#   --batch_size_val $batch_size \
#   --num_sampling_level $num_sampling_level \
#   --regress_position_offset $regress_position_offset \
#   --accumulate_grad_batches $accumulate_grad_batches \
#   --gripper_loc_bounds_file $gripper_loc_bounds_file \
#   --variations {0..199} \
#   --use_instruction $use_instruction \
#   --run_log_dir $task-WITH-INSTRUCTION
#done



main_dir=03_10_improve_pick
num_workers=2
batch_size=3
accumulate_grad_batches=2
gripper_loc_bounds_file=tasks/10_autolambda_tasks_location_bounds.json
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3

num_sampling_level=2
regress_position_offset=1
for task in pick_and_lift pick_up_cup; do
  sbatch train_1gpu_12gb.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --num_workers $num_workers \
   --batch_size $batch_size \
   --batch_size_val $batch_size \
   --num_sampling_level $num_sampling_level \
   --regress_position_offset $regress_position_offset \
   --accumulate_grad_batches $accumulate_grad_batches \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --run_log_dir $task-two-levels-with-offset
done

num_sampling_level=3
regress_position_offset=0
for task in pick_and_lift pick_up_cup; do
  sbatch train_1gpu_12gb.sh \
   --tasks $task \
   --dataset $dataset \
   --valset $valset \
   --exp_log_dir $main_dir \
   --num_workers $num_workers \
   --batch_size_val $batch_size \
   --num_sampling_level $num_sampling_level \
   --regress_position_offset $regress_position_offset \
   --accumulate_grad_batches $accumulate_grad_batches \
   --gripper_loc_bounds_file $gripper_loc_bounds_file \
   --run_log_dir $task-three-levels-no-offset
done
