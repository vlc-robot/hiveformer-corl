#!/bin/sh

root=/home/tgervet
train_seed=0
val_seed=1
task_file=10_autolambda_tasks.csv
output_dir=$root/datasets/hiveformer/packaged

#experiment=ghost_points_exp2
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_32gb.sh \
#    --exp_log_dir $experiment \
#    --run_log_dir $task \
#    --tasks $task \
#    --dataset $output_dir/$train_seed \
#    --valset $output_dir/$val_seed \
#    --model develop \
#    --batch_size 10 \
#    --lr 0.001
#done

task_file=debugging_tasks.csv
experiment=debug_mask2former_exp1
for task in $(cat $task_file | tr '\n' ' '); do
  for lr in 0.00005 0.0005; do
    sbatch train_1gpu.sh \
      --exp_log_dir $experiment \
      --tasks $task \
      --run_log_dir "mask2former_$task_$lr" \
      --dataset $output_dir/$train_seed \
      --valset $output_dir/$val_seed \
      --lr $lr \
      --model develop
  done
done
for task in $(cat $task_file | tr '\n' ' '); do
    sbatch train_1gpu.sh \
      --exp_log_dir $experiment \
      --tasks $task \
      --run_log_dir "original_$task" \
      --dataset $output_dir/$train_seed \
      --valset $output_dir/$val_seed \
      --model original
done
