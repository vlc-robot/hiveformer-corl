#!/bin/sh

main_dir=02_27_multi_task
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3

for task in put_knife_on_chopping_board; do
  for use_instruction in 0 1; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --exp_log_dir $main_dir \
       --use_instruction $use_instruction \
       --num_workers 3 \
       --run_log_dir $task-use-instruction-$use_instruction
  done
done

task_file=tasks/7_interesting_tasks.csv
for use_instruction in 0 1; do
  sbatch train_1gpu_32gb.sh \
     --tasks $(cat $task_file | tr '\n' ' ') \
     --dataset $dataset \
     --valset $valset \
     --exp_log_dir $main_dir \
     --use_instruction $use_instruction \
     --num_workers 3 \
     --run_log_dir MULTITASK-use-instruction-$use_instruction
done
