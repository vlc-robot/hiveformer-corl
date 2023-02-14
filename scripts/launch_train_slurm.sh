#!/bin/sh

#main_dir=reproduce_hiveformer
#task_file=tasks/10_autolambda_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --exp_log_dir $main_dir \
#     --run_log_dir HIVEFORMER-$task \
#     --model original \
#     --batch_size 32 \
#     --train_iters 100_000
#done

main_dir=02_14_tune_ghost_points
dataset=/home/tgervet/datasets/hiveformer/packaged/0
valset=/home/tgervet/datasets/hiveformer/packaged/1
image_size="128,128"

batch_size=25
num_ghost_points=1000
for task in put_money_in_safe; do
  for fine_sampling_cube_size in 0.08 0.16 0.04; do
    for use_ground_truth_position_for_sampling_val in 0 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --valset $valset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --run_log_dir $task \
         --batch_size $batch_size \
         --fine_sampling_cube_size $fine_sampling_cube_size \
         --num_ghost_points $num_ghost_points \
         --use_ground_truth_position_for_sampling_val $use_ground_truth_position_for_sampling_val
    done
  done
done

batch_size=15
num_ghost_points=2000
for task in put_money_in_safe; do
  for fine_sampling_cube_size in 0.08 0.16 0.04; do
    for use_ground_truth_position_for_sampling_val in 0 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --valset $valset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --run_log_dir $task-cube-$fine_sampling_cube_size-ghost-$num_ghost_points-gtsample-$use_ground_truth_position_for_sampling_val \
         --batch_size $batch_size \
         --fine_sampling_cube_size $fine_sampling_cube_size \
         --num_ghost_points $num_ghost_points \
         --use_ground_truth_position_for_sampling_val $use_ground_truth_position_for_sampling_val
    done
  done
done

#task_file=tasks/2_debugging_tasks.csv
#dataset=/home/tgervet/datasets/hiveformer/packaged/2
#valset=/home/tgervet/datasets/hiveformer/packaged/3
#image_size="256,256"
#batch_size=20
#for task in $(cat $task_file | tr '\n' ' '); do
#  for randomize_ground_truth_ghost_point in 0 1; do
#      sbatch train_1gpu_32gb.sh \
#         --tasks $task \
#         --dataset $dataset \
#         --valset $valset \
#         --image_size $image_size \
#         --exp_log_dir $main_dir \
#         --run_log_dir $task-img-$image_size-rand-$randomize_ground_truth_ghost_point \
#         --batch_size $batch_size \
#         --randomize_ground_truth_ghost_point $randomize_ground_truth_ghost_point \
#         --num_workers 2
#    done
#done
