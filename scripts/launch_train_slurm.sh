#!/bin/sh

main_dir=02_23_analogical_poc

for support_set in "self" "rest_of_batch"; do
  for task in put_money_in_safe; do
    sbatch train_1gpu_32gb.sh \
       --tasks $task \
       --dataset /home/tgervet/datasets/hiveformer/packaged/2 \
       --valset /home/tgervet/datasets/hiveformer/packaged/3 \
       --image_size "256,256" \
       --exp_log_dir $main_dir \
       --model analogical \
       --batch_size 15 \
       --num_workers 2 \
       --position_prediction_only 1 \
       --rotation_parametrization quat_from_top_ghost \
       --support_set $support_set \
       --run_log_dir $task-$support_set
  done
done
