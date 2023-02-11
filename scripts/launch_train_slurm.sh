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
#
#task_file=tasks/2_debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --exp_log_dir $main_dir \
#     --run_log_dir BASELINE-$task \
#     --model baseline
#done

main_dir=02_10_coarse_to_fine

task_file=tasks/2_debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  for size in 0.1 0.05; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
       --valset /home/tgervet/datasets/hiveformer/packaged/1 \
       --exp_log_dir $main_dir \
       --run_log_dir COARSE2FINE-$task \
       --fine_sampling_cube_size $size \
       --coarse_to_fine_sampling 1
  done
done

task_file=tasks/2_debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir COARSE-$task \
     --coarse_to_fine_sampling 0
done