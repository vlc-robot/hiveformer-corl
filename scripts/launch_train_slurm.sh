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

main_dir=02_16_regress_position_offset4
dataset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
fine_sampling_cube_size=0.08
num_workers=2
task=put_money_in_safe
num_ghost_points=2000
batch_size=12

sbatch train_1gpu_32gb.sh \
     --tasks $task \
     --dataset $dataset \
     --image_size $image_size \
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --num_workers $num_workers \
     --num_ghost_points $num_ghost_points \
     --regress_position_offset 0 \
     --run_log_dir OVERFIT-$task-NO-OFFSET

for position_offset_loss_coeff in 10 100 1000; do
  for points_supervised_for_offset in all fine closest; do
    sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --batch_size $batch_size \
         --num_workers $num_workers \
         --num_ghost_points $num_ghost_points \
         --regress_position_offset 1 \
         --position_offset_loss_coeff $position_offset_loss_coeff \
         --points_supervised_for_offset $points_supervised_for_offset \
         --run_log_dir OVERFIT-$task-$position_offset_loss_coeff-$points_supervised_for_offset
  done
done

#main_dir=02_15_tune_ghost_points
#dataset=/home/tgervet/datasets/hiveformer/packaged/3
#image_size="256,256"
#fine_sampling_cube_size=0.08
#use_ground_truth_position_for_sampling_val=0
#num_workers=2
#task=put_money_in_safe
#num_ghost_points=2000
#batch_size=12
#
#sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset $dataset \
#     --image_size $image_size \
#     --exp_log_dir $main_dir \
#     --batch_size $batch_size \
#     --num_workers $num_workers \
#     --num_ghost_points $num_ghost_points \
#     --run_log_dir $task-NOLABELSMOOTHING \
#     --label_smoothing 0
#
#sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset $dataset \
#     --image_size $image_size \
#     --exp_log_dir $main_dir \
#     --batch_size $batch_size \
#     --num_workers $num_workers \
#     --num_ghost_points $num_ghost_points \
#     --run_log_dir $task-LOSSATALLLAYERS \
#     --compute_loss_at_all_layers 1
#
#sbatch train_1gpu_32gb.sh \
#     --tasks $task \
#     --dataset $dataset \
#     --image_size $image_size \
#     --exp_log_dir $main_dir \
#     --batch_size $batch_size \
#     --num_workers $num_workers \
#     --num_ghost_points $num_ghost_points \
#     --run_log_dir $task-BIGGERLR \
#     --lr 5e-4
#
#for num_ghost_points in 500 1000 2000; do
#  sbatch train_1gpu_32gb.sh \
#       --tasks $task \
#       --dataset $dataset \
#       --image_size $image_size \
#       --exp_log_dir $main_dir \
#       --batch_size $batch_size \
#       --num_workers $num_workers \
#       --num_ghost_points $num_ghost_points \
#       --run_log_dir $task-$num_ghost_points-ghost_points
#done
