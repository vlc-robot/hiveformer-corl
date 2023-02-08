#!/bin/sh

main_dir=02_07_relative_pe

#task_file=debugging_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  for loss in mse bce ce; do
#    sbatch train_1gpu_32gb.sh \
#       --tasks $task \
#       --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#       --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#       --exp_log_dir $main_dir \
#       --run_log_dir $task-$loss \
#       --position_loss $loss \
#       --relative_attention 1 \
#       --embedding_dim 120
#  done
#done

task_file=debugging_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir $task-small \
     --position_loss ce \
     --relative_attention 1 \
     --embedding_dim 60 \
     --num_ghost_point_cross_attn_layers 2 \
     --num_query_cross_attn_layers 2
done

task_file=more_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
     --exp_log_dir $main_dir \
     --run_log_dir $task-medium \
     --position_loss ce \
     --relative_attention 1 \
     --embedding_dim 120 \
     --num_ghost_point_cross_attn_layers 4 \
     --num_query_cross_attn_layers 4
done
