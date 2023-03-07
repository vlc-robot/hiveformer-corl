#!/bin/sh

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
main_dir=exp

# --------------------------------------------------------------------------------------------
# Training configurations
# --------------------------------------------------------------------------------------------
# Single task
#num_workers=2
#batch_size=3
#accumulate_grad_batches=2
#task_file=tasks/7_interesting_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#   --tasks $task \
#   --dataset $dataset \
#   --valset $valset \
#   --exp_log_dir $main_dir \
#   --num_workers $num_workers \
#   --batch_size $batch_size \
#   --accumulate_grad_batches $accumulate_grad_batches \
#   --run_log_dir $task
#done

# Multi task
#embedding_dim=120
#num_workers=12
#batch_size=16
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#sbatch train_4gpu_12gb.sh \
#  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#  --tasks $(cat $task_file | tr '\n' ' ') \
#  --dataset $dataset \
#  --valset $valset \
#  --exp_log_dir $main_dir \
#  --num_workers $num_workers \
#  --batch_size $batch_size \
#  --train_iters $train_iters \
#  --embedding_dim $embedding_dim \
#  --run_log_dir BASELINE-MULTI-TASK

# Analogy
#embedding_dim=120
#num_workers=12
#batch_size=8
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#sbatch train_4gpu_12gb.sh \
#  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#  --model analogical \
#  --rotation_parametrization "quat_from_top_ghost" \
#  --support_set "others" \
#  --tasks $(cat $task_file | tr '\n' ' ') \
#  --dataset $dataset \
#  --valset $valset \
#  --exp_log_dir $main_dir \
#  --num_workers $num_workers \
#  --batch_size $batch_size \
#  --train_iters $train_iters \
#  --embedding_dim $embedding_dim \
#  --run_log_dir ANALOGY-MULTI-TASK
# --------------------------------------------------------------------------------------------

#main_dir=03_06_task_specific_biases2
#embedding_dim=120
#num_workers=12
#train_iters=500_000
#task_file=tasks/7_interesting_tasks.csv
#
#batch_size=8
#for task_specific_biases in 0 1; do
#  sbatch train_4gpu_12gb.sh \
#    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#    --model analogical \
#    --rotation_parametrization "quat_from_top_ghost" \
#    --support_set "others" \
#    --tasks $(cat $task_file | tr '\n' ' ') \
#    --dataset $dataset \
#    --valset $valset \
#    --exp_log_dir $main_dir \
#    --num_workers $num_workers \
#    --batch_size $batch_size \
#    --train_iters $train_iters \
#    --embedding_dim $embedding_dim \
#    --run_log_dir ANALOGY-task_specific_biases-$task_specific_biases
#done
#
#batch_size=16
#for task_specific_biases in 0 1; do
#  sbatch train_4gpu_12gb.sh \
#    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#    --model baseline \
#    --tasks $(cat $task_file | tr '\n' ' ') \
#    --dataset $dataset \
#    --valset $valset \
#    --exp_log_dir $main_dir \
#    --num_workers $num_workers \
#    --batch_size $batch_size \
#    --train_iters $train_iters \
#    --embedding_dim $embedding_dim \
#    --run_log_dir BASELINE-task_specific_biases-$task_specific_biases
#done

main_dir=03_07_hiveformer_10_demos
task_file=tasks/7_interesting_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_12gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --image_size "128,128" \
     --exp_log_dir $main_dir \
     --model original \
     --max_episodes_per_taskvar 10 \
     --run_log_dir $task
done
