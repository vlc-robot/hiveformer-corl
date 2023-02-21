#!/bin/sh

main_dir=02_20_compare_hiveformer_and_baseline
task_file=tasks/10_autolambda_tasks.csv

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_12gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --image_size "128,128" \
     --exp_log_dir $main_dir \
     --model original \
     --run_log_dir HIVEFORMER-$task
done

for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
     --valset /home/tgervet/datasets/hiveformer/packaged/3 \
     --image_size "256,256" \
     --exp_log_dir $main_dir \
     --model baseline \
     --batch_size 14 \
     --num_workers 3 \
     --run_log_dir BASELINE-$task
done
