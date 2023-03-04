#!/bin/sh

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
num_workers=2
batch_size=4
accumulate_grad_batches=2

main_dir=03_05_baseline_single_task_vs_multi_task
task_file=tasks/7_interesting_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  sbatch train_1gpu_12gb.sh \
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --exp_log_dir $main_dir \
     --num_workers $num_workers \
     --batch_size $batch_size \
     --accumulate_grad_batches $accumulate_grad_batches \
     --run_log_dir $task
done


main_dir=03_05_analogy_single_task
task_file=tasks/2_easy_tasks.csv
for task in $(cat $task_file | tr '\n' ' '); do
  for support_set in self others; do
    for global_correspondence in 0; do
      sbatch train_1gpu_12gb.sh \
         --tasks $task \
         --rotation_parametrization "quat_from_top_ghost" \
         --model analogical \
         --dataset $dataset \
         --valset $valset \
         --exp_log_dir $main_dir \
         --num_workers $num_workers \
         --batch_size $batch_size \
         --support_set $support_set \
         --global_correspondence $global_correspondence \
         --accumulate_grad_batches $accumulate_grad_batches \
         --run_log_dir $task-support_set-$support_set-global_correspondence-$global_correspondence
    done
  done
done


#main_dir=03_03_MULTI_TASK
#dataset=/home/tgervet/datasets/hiveformer/packaged/2
#valset=/home/tgervet/datasets/hiveformer/packaged/3
#task_file=tasks/7_interesting_tasks.csv
#embedding_dim=120
#lr=1e-4
#train_iters=500_000
#num_workers=16

# Multi-task
#batch_size=8
#model=baseline
#for accumulate_grad_batches in 4; do
#  for use_instruction in 1; do
#    for backbone in clip; do
#      sbatch train_4gpu_12gb.sh \
#         --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#         --model $model \
#         --tasks $(cat $task_file | tr '\n' ' ') \
#         --lr $lr \
#         --dataset $dataset \
#         --valset $valset \
#         --exp_log_dir $main_dir \
#         --num_workers $num_workers \
#         --batch_size $batch_size \
#         --train_iters $train_iters \
#         --embedding_dim $embedding_dim \
#         --use_instruction $use_instruction \
#         --backbone $backbone \
#         --accumulate_grad_batches $accumulate_grad_batches \
#         --run_log_dir BASELINE-INSTRUCTION-EVERYWHERE-instr-$use_instruction-backbone-$backbone-acc-$accumulate_grad_batches
#    done
#  done
#done

# Analogy
#batch_size=4
#model=analogical
#use_instruction=1
#backbone=clip
#for accumulate_grad_batches in 4; do
#  for support_set in others; do
#    for global_correspondence in 0; do
#      sbatch train_4gpu_12gb.sh \
#         --devices cuda:0 cuda:1 cuda:2 cuda:3 \
#         --rotation_parametrization "quat_from_top_ghost" \
#         --model $model \
#         --tasks $(cat $task_file | tr '\n' ' ') \
#         --lr $lr \
#         --dataset $dataset \
#         --valset $valset \
#         --exp_log_dir $main_dir \
#         --num_workers $num_workers \
#         --batch_size $batch_size \
#         --train_iters $train_iters \
#         --embedding_dim $embedding_dim \
#         --support_set $support_set \
#         --global_correspondence $global_correspondence \
#         --accumulate_grad_batches $accumulate_grad_batches \
#         --use_instruction $use_instruction \
#         --backbone $backbone \
#         --run_log_dir ANALOGICAL-INSTRUCTION-EVERYWHERE-support_set-$support_set-global_correspondence-$global_correspondence-acc-$accumulate_grad_batches
#    done
#  done
#done
