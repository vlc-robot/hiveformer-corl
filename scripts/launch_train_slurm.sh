#!/bin/sh

#main_dir=reproduce_hiveformer
#task_file=tasks/10_autolambda_tasks.csv
#for task in $(cat $task_file | tr '\n' ' '); do
#  sbatch train_1gpu_12gb.sh \
#     --tasks $task \
#     --dataset /home/tgervet/datasets/hiveformer/packaged/0 \
#     --valset /home/tgervet/datasets/hiveformer/packaged/1 \
#     --exp_log_dir $main_dir \
#     --run_log_dir HIVEFORMER-$task \
#     --model original \
#     --batch_size 32 \
#     --train_iters 100_000
#done

main_dir=02_14_tune_hyperparams

#task_file=tasks/2_debugging_tasks.csv
#dataset=/home/tgervet/datasets/hiveformer/packaged/2
#valset=/home/tgervet/datasets/hiveformer/packaged/3
#image_size="256,256"
#batch_size=20
#for task in $(cat $task_file | tr '\n' ' '); do
#  for randomize_ground_truth_ghost_point in 0 1; do
#      sbatch train_1gpu_32gb.sh \
#         --tasks $task \
#         --dataset $dataset \
#         --valset $valset \
#         --image_size $image_size \
#         --exp_log_dir $main_dir \
#         --run_log_dir $task-img-$image_size-rand-$randomize_ground_truth_ghost_point \
#         --batch_size $batch_size \
#         --randomize_ground_truth_ghost_point $randomize_ground_truth_ghost_point \
#         --num_workers 2
#    done
#done

dataset=/home/tgervet/datasets/hiveformer/packaged/0
valset=/home/tgervet/datasets/hiveformer/packaged/1
image_size="128,128"
batch_size=25
for task in put_money_in_safe; do
  for lr in 5e-5 5e-4; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --image_size $image_size \
       --exp_log_dir $main_dir \
       --run_log_dir $task-img-$image_size-lr-$lr \
       --batch_size $batch_size \
       --lr $lr
  done
done

for task in put_money_in_safe; do
  for fine_sampling_cube_size in 0.08 0.16; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --image_size $image_size \
       --exp_log_dir $main_dir \
       --run_log_dir $task-img-$image_size-cube-$fine_sampling_cube_size \
       --batch_size $batch_size \
       --fine_sampling_cube_size $fine_sampling_cube_size
  done
done

for task in put_money_in_safe; do
  sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --run_log_dir $task-img-$image_size-BIG-MODEL \
     --batch_size 12 \
     --embedding_dim 120 \
     --num_ghost_point_cross_attn_layers 4 \
     --num_ghost_point_cross_attn_layers 4
done

sbatch train_1gpu_32gb.sh \
     --tasks put_money_in_safe \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --run_log_dir $task-img-$image_size-BCE \
     --batch_size $batch_size \
     --position_loss bce

sbatch train_1gpu_32gb.sh \
     --tasks put_money_in_safe \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --run_log_dir $task-img-$image_size-LOSSLASTLAYER \
     --batch_size $batch_size \
     --compute_loss_at_all_layers 0

sbatch train_1gpu_32gb.sh \
     --tasks put_money_in_safe \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --run_log_dir $task-img-$image_size-NOLABELSMOOTHING \
     --batch_size $batch_size \
     --label_smoothing 0