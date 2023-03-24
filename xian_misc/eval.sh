valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val/
# valset=/home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val_fail/
# valset=/home/zhouxian/git/datasets/hiveformer/packaged/3
task=reach_target
task=push_button
task=slide_block_to_target
task=pick_up_cup
task=take_umbrella_out_of_umbrella_stand
task=pick_and_lift
task=put_knife_on_chopping_board
task=take_money_out_safe
task=put_money_in_safe
# task=stack_wine

gripper_bounds_buffer=0.04
use_instruction=0
weight_tying=1

gp_emb_tying=1
simplify=1
num_sampling_level=3
regress_position_offset=0
num_ghost_points_val=20000

python eval.py\
     --tasks $task\
     --checkpoint /home/zhouxian/git/hiveformer/train_logs/03_21_10demo/put_money_in_safe-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo10-lr1e-4-seed0_version159632/model.step=95000-value=0.00000.pth \
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
     # --max_episodes 50
