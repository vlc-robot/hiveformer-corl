#!/bin/sh

main_dir=03_03_MULTI_TASK
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
task_file=tasks/7_interesting_tasks.csv
embedding_dim=120
lr=1e-4
train_iters=500_000
num_workers=16

# Multi-task
batch_size=8
model=baseline
for use_instruction in 0; do
  for backbone in resnet; do
    sbatch train_4gpu_12gb.sh \
       --devices cuda:0 cuda:1 cuda:2 cuda:3 \
       --model $model \
       --tasks $(cat $task_file | tr '\n' ' ') \
       --lr $lr \
       --dataset $dataset \
       --valset $valset \
       --exp_log_dir $main_dir \
       --num_workers $num_workers \
       --batch_size $batch_size \
       --train_iters $train_iters \
       --embedding_dim $embedding_dim \
       --use_instruction $use_instruction \
       --backbone $backbone \
       --run_log_dir BASELINE-instr-$use_instruction-backbone-$backbone
  done
done

# Analogy
batch_size=4
model=analogical
for support_set in self; do
  for global_correspondence in 0; do
    sbatch train_4gpu_12gb.sh \
       --devices cuda:0 cuda:1 cuda:2 cuda:3 \
       --rotation_parametrization "quat_from_top_ghost" \
       --model $model \
       --tasks $(cat $task_file | tr '\n' ' ') \
       --lr $lr \
       --dataset $dataset \
       --valset $valset \
       --exp_log_dir $main_dir \
       --num_workers $num_workers \
       --batch_size $batch_size \
       --train_iters $train_iters \
       --embedding_dim $embedding_dim \
       --support_set $support_set \
       --global_correspondence $global_correspondence \
       --run_log_dir ANALOGICAL-support_set-$support_set-global_correspondence-$global_correspondence
  done
done
