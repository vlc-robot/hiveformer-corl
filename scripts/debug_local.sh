# HIVEFORMER
exp=03_24_hiveformer_setting
ckpts=(
  insert_onto_square_peg-HIVEFORMER_version161659
)
tasks=(
  insert_onto_square_peg
)

data_dir=/home/sirdome/katefgroup/raw/74_hiveformer_tasks_val
num_episodes=10
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
use_instruction=0
num_ghost_points=10000
headless=0
offline=0

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points \
    --output_file /home/sirdome/katefgroup/hiveformer/eval_new.json
#    --variations {0..60}
done
