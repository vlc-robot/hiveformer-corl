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

main_dir=02_17_train_rotation2
dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
image_size="256,256"
num_workers=2
task=put_money_in_safe
batch_size=20

for rotation_parametrization in quat_from_query; do
  for rotation_loss_coeff in 10000; do
    sbatch train_1gpu_32gb.sh \
         --tasks $task \
         --dataset $dataset \
         --valset $dataset \
         --image_size $image_size \
         --exp_log_dir $main_dir \
         --batch_size $batch_size \
         --num_workers $num_workers \
         --run_log_dir $task-$rotation_loss_coeff-$rotation_parametrization \
         --rotation_parametrization $rotation_parametrization \
         --rotation_loss_coeff $rotation_loss_coeff
  done
done
