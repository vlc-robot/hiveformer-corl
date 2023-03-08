
# main_dir=02_27_rotation
# main_dir=02_28_rotation_0.1
# main_dir=02_28_rotation_0.1_quatloss
# main_dir=02_28_rotation_0.1_att
# main_dir=02_28_rotation_0.1_att_debug
# main_dir=02_28_rotation_0.1_att_nonepe
# main_dir=02_28_rotation_0.1_att_noatt
# main_dir=03_03_multi_level_sampling
# main_dir=03_04_multi_level_sampling
main_dir=03_07_test

dataset=/home/tgervet/datasets/hiveformer/packaged/2
valset=/home/tgervet/datasets/hiveformer/packaged/3
dataset=/home/zhouxian/git/datasets/hiveformer/packaged/2
valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
task=put_money_in_safe
batch_size=16
# batch_size=8
batch_size=4
lr=1e-4


num_sampling_level=3
regress_position_offset=0
num_ghost_points=1500
gripper_bounds_buffer=0.04
use_instruction=0


python train.py\
     --tasks $task \
     --dataset $dataset \
     --valset $valset \
     --val_freq 10\
     --exp_log_dir $main_dir \
     --batch_size $batch_size \
     --use_instruction $use_instruction\
     --num_ghost_points $num_ghost_points\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --lr $lr\
     --run_log_dir $task-offset$regress_position_offset-B$batch_size-lr$lr
     # --run_log_dir $task-$rotation_parametrization-offset$regress_position_offset-N$num_sampling_level-P$num_ghost_points-EP$max_episodes_per_taskvar-B$batch_size-lr$lr

