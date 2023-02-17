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

main_dir=02_17_train_with_offset2
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
num_workers=2
task=put_money_in_safe
num_ghost_points=2000
batch_size=12

sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --num_workers $num_workers \
     --num_ghost_points $num_ghost_points \
     --run_log_dir $task-NOOFFSET \
     --regress_position_offset 0

sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --num_workers $num_workers \
     --num_ghost_points $num_ghost_points \
     --run_log_dir $task-NOLABELSMOOTHING \
     --label_smoothing 0

sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --num_workers $num_workers \
     --num_ghost_points $num_ghost_points \
     --run_log_dir $task-LOSSATALLLAYERS \
     --compute_loss_at_all_layers 1

sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --num_workers $num_workers \
     --num_ghost_points $num_ghost_points \
     --run_log_dir $task-BIGGERLR \
     --lr 5e-4

for num_ghost_points in 500 1000 2000; do
  sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset $dataset \
       --valset $valset \
       --image_size $image_size \
       --exp_log_dir $main_dir \
       --batch_size $batch_size \
       --num_workers $num_workers \
       --num_ghost_points $num_ghost_points \
       --run_log_dir $task-$num_ghost_points-ghost_points
done
