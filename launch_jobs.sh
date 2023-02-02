#!/bin/sh

#task_file=10_autolambda_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --batch_size 10 \
#     --exp_log_dir train_cross_entropy_loss \
#     --run_log_dir $task \
#     --compute_loss_at_all_layers 0
#done

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --batch_size 10 \
     --exp_log_dir train_cross_entropy_loss_with_intermediate_losses \
     --run_log_dir $task \
     --compute_loss_at_all_layers 1
done
