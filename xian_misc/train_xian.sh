
# main_dir=02_27_rotation
# main_dir=02_28_rotation_0.1
# main_dir=02_28_rotation_0.1_quatloss
# main_dir=02_28_rotation_0.1_att
# main_dir=02_28_rotation_0.1_att_debug
# main_dir=02_28_rotation_0.1_att_nonepe
# main_dir=02_28_rotation_0.1_att_noatt
# main_dir=03_03_multi_level_sampling
# main_dir=03_04_multi_level_sampling
# main_dir=03_09_ablations
# main_dir=03_10_dense_val_sampling
# main_dir=03_13
# main_dir=03_14_debug_offset0
main_dir=03_16_debugdebugdebug

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
# dataset=/home/zhouxian/git/datasets/hiveformer/packaged/2
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
# task=put_money_in_safe
task=pick_and_lift
batch_size_val=4
batch_size=16
# batch_size=32
# batch_size=8
# batch_size=2
lr=1e-4

gripper_bounds_buffer=0.04
use_instruction=0
weight_tying=1
max_episodes_per_taskvar=100
symmetric_rotation_loss=0
num_ghost_points=1000
num_ghost_points_val=10000

gp_emb_tying=1
simplify=1
num_sampling_level=3
regress_position_offset=1


python train.py\
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --simplify $simplify\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --batch_size_val $batch_size_val \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --num_ghost_points_val $num_ghost_points_val\
     --max_episodes_per_taskvar $max_episodes_per_taskvar\
     --symmetric_rotation_loss $symmetric_rotation_loss\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --lr $lr\
     --run_log_dir $task-offset$regress_position_offset-N$num_sampling_level-T$num_ghost_points-V$num_ghost_points_val-symrot$symmetric_rotation_loss-gptie$gp_emb_tying-simp$simplify-B$batch_size-lr$lr

