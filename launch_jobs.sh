#!/bin/sh

main_dir=02_09_num_ghost_points

#task_file=debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  for spread in 0.01 0.001; do
#    for sample in 0 1; do
#      sbatch train_1gpu_32gb.sh \
#         --tasks $task \
#         --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#         --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#         --exp_log_dir $main_dir \
#         --run_log_dir $task-spread-$spread-sample-$sample \
#         --ground_truth_gaussian_spread $spread \
#         --use_ground_truth_position_for_sampling $sample
#    done
#  done
#done

#task_file=debugging_tasks.csv
task_file=more_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  for n in 125 250 500 1000; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
       --valset /home/tgervet/datasets/hiveformer/packaged/1 \
       --exp_log_dir $main_dir \
       --run_log_dir $task-$n \
       --num_ghost_points $n
  done
done