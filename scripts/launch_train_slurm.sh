#!/bin/sh

main_dir=reproduce_hiveformer

task_file=tasks/10_autolambda_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_12gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir HIVEFORMER-$task \
     --model original \
     --batch_size 32 \
     --train_iters 100_000
done

task_file=tasks/2_debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir BASELINE-$task \
     --model baseline
done