#!/bin/sh

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --batch_size 10 \
     --exp_log_dir xentropy \
     --run_log_dir $task \
     --compute_loss_at_all_layers 0
done

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --batch_size 10 \
     --exp_log_dir xentropy_intermediate_losses \
     --run_log_dir $task \
     --compute_loss_at_all_layers 1
done
