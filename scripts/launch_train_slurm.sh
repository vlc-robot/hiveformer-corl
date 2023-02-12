#!/bin/sh

#main_dir=reproduce_hiveformer
#
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

#main_dir=02_11_coarse_to_fine

#task_file=tasks/2_debugging_tasks.csv
#dataset=/home/tgervet/datasets/hiveformer/packaged/0
#valset=/home/tgervet/datasets/hiveformer/packaged/1
#image_size="128,128"
#for task in $(cat $task_file | tr '\n' ' '); do
#  for c2f in 0 1; do
#    sbatch train_1gpu_32gb.sh \
#       --tasks $task \
#       --dataset $dataset \
#       --valset $valset \
#       --image_size $image_size \
#       --exp_log_dir $main_dir \
#       --run_log_dir IMAGE-$image_size-C2F-$c2f-$task \
#       --coarse_to_fine_sampling $c2f
#  done
#done

#task_file=tasks/2_debugging_tasks.csv
#dataset=/home/tgervet/datasets/hiveformer/packaged/2
#valset=/home/tgervet/datasets/hiveformer/packaged/3
#image_size="256,256"
#for task in $(cat $task_file | tr '\n' ' '); do
#  for c2f in 1; do
#    sbatch train_1gpu_32gb.sh \
#       --tasks $task \
#       --dataset $dataset \
#       --valset $valset \
#       --image_size $image_size \
#       --exp_log_dir $main_dir \
#       --run_log_dir IMAGE-$image_size-C2F-$c2f-$task \
#       --coarse_to_fine_sampling $c2f \
#       --num_workers 2
#  done
#done

main_dir=02_12_overfit_coarse_to_fine2

#task_file=tasks/2_debugging_tasks.csv
#dataset=/home/tgervet/datasets/hiveformer/packaged/1
#image_size="128,128"
#for task in $(cat $task_file | tr '\n' ' '); do
#  for ground_truth_gaussian_spread in 0.01 0.001; do
#    for use_ground_truth_position_for_sampling in 1; do
#      sbatch train_1gpu_32gb.sh \
#         --tasks $task \
#         --dataset $dataset \
#         --image_size $image_size \
#         --exp_log_dir $main_dir \
#         --run_log_dir $task-gtspread-$ground_truth_gaussian_spread-gtsample-$use_ground_truth_position_for_sampling-IMPERFECT \
#         --coarse_to_fine_sampling 1 \
#         --batch_size 25 \
#         --ground_truth_gaussian_spread $ground_truth_gaussian_spread \
#         --use_ground_truth_position_for_sampling $use_ground_truth_position_for_sampling
#    done
#  done
#done

task_file=tasks/2_debugging_tasks.csv
dataset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
#for task in $(cat $task_file | tr '\n' ' '); do
for task in put_money_in_safe; do
  for ground_truth_gaussian_spread in 0.01 0.001; do
    for use_ground_truth_position_for_sampling in 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --run_log_dir $task-gtspread-$ground_truth_gaussian_spread-gtsample-$use_ground_truth_position_for_sampling-IMPERFECT-256x256 \
         --coarse_to_fine_sampling 1 \
         --batch_size 10 \
         --ground_truth_gaussian_spread $ground_truth_gaussian_spread \
         --use_ground_truth_position_for_sampling $use_ground_truth_position_for_sampling
    done
  done
done
