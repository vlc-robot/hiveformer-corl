#!/bin/sh

main_dir=02_28_multi_gpu_multi_task
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3

# Big model with big batch size on big GPUs
#task_file=tasks/7_interesting_tasks.csv
#for use_instruction in 1; do
#  sbatch train_4gpu_32gb.sh \
#     --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#     --tasks $(cat $task_file | tr '\n' ' ') \
#     --dataset $dataset \
#     --valset $valset \
#     --exp_log_dir $main_dir \
#     --use_instruction $use_instruction \
#     --num_workers 16 \
#     --batch_size 32 \
#     --embedding_dim 120 \
#     --lr 1e-4 \
#     --run_log_dir LARGE-MULTITASK-use-instruction-$use_instruction
#done

# Small model with and without and without instructions on small GPUs
#task_file=tasks/7_interesting_tasks.csv
#for use_instruction in 0 1; do
#  sbatch train_4gpu_12gb.sh \
#     --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#     --tasks $(cat $task_file | tr '\n' ' ') \
#     --dataset $dataset \
#     --valset $valset \
#     --exp_log_dir $main_dir \
#     --use_instruction $use_instruction \
#     --num_workers 12 \
#     --batch_size 16 \
#     --run_log_dir SMALL-MULTITASK-use-instruction-$use_instruction
#done

# Check that there were no regressions with using a large workspace
main_dir=02_28_check_for_regressions
for task in put_knife_on_chopping_board take_money_out_safe; do
  for num_ghost_points in 1000; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --exp_log_dir $main_dir \
       --num_ghost_points $num_ghost_points \
       --num_workers 3 \
       --batch_size 16 \
       --run_log_dir SANITY-CHECK-$task-ghost-$num_ghost_points
  done
done
