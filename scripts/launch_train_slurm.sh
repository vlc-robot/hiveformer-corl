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

main_dir=02_12_overfit_coarse_to_fine6

task_file=tasks/2_debugging_tasks.csv
dataset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
batch_size=20
for task in $(cat $task_file | tr '\n' ' '); do
  for randomize_ground_truth_ghost_point in 0; do
    for separate_coarse_and_fine_losses in 0 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --run_log_dir $task-img-$image_size-rand-$randomize_ground_truth_ghost_point-sep-$separate_coarse_and_fine_losses \
         --batch_size $batch_size \
         --randomize_ground_truth_ghost_point $randomize_ground_truth_ghost_point \
         --separate_coarse_and_fine_losses $separate_coarse_and_fine_losses
    done
  done
done

task_file=tasks/2_debugging_tasks.csv
dataset=/home/tgervet/datasets/hiveformer/packaged/1
image_size="128,128"
batch_size=25
for task in $(cat $task_file | tr '\n' ' '); do
  for randomize_ground_truth_ghost_point in 0; do
    for separate_coarse_and_fine_losses in 0 1; do
      sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --run_log_dir $task-img-$image_size-rand-$randomize_ground_truth_ghost_point-sep-$separate_coarse_and_fine_losses \
         --batch_size $batch_size \
         --randomize_ground_truth_ghost_point $randomize_ground_truth_ghost_point \
         --separate_coarse_and_fine_losses $separate_coarse_and_fine_losses
    done
  done
done
