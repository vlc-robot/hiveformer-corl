valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val/
# valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val_fail/
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
task=put_money_in_safe
task=pick_and_lift

gripper_bounds_buffer=0.04
use_instruction=0
weight_tying=1

gp_emb_tying=0
simplify=0
num_sampling_level=3
regress_position_offset=1
num_ghost_points_val=10000

python eval.py\
     --tasks $task\
     --checkpoint /home/zhouxian/git/hiveformer/train_logs/03_13/pick_and_lift-offset1-N3-T1000-V10000-symrot0-B16-lr1e-4_version157395/model.step=160000-value=0.00000.pth \
     --data_dir $valset\
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --simplify $simplify\
     --image_size 256,256\
     --offline 0\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --num_ghost_points_val $num_ghost_points_val\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --run_log_dir $task-ONLINE\
