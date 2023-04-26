# HIVEFORMER
exp=03_24_hiveformer_setting
ckpts=(
  close_door-HIVEFORMER_version161046
  take_usb_out_of_computer-HIVEFORMER_version163526
  place_shape_in_shape_sorter-HIVEFORMER_version163673
  sweep_to_dustpan-HIVEFORMER_version163674
  take_plate_off_colored_dish_rack-HIVEFORMER_version163675
  place_hanger_on_rack-HIVEFORMER_version163676
  plug_charger_in_power_supply-HIVEFORMER_version163667
  reach_and_drag-HIVEFORMER_version163669
  setup_checkers-HIVEFORMER_version163670
  tower3-HIVEFORMER_version162809
  straighten_rope-HIVEFORMER_version163672
  wipe_desk-HIVEFORMER_version164479
  change_channel-HIVEFORMER_version165805
  tv_on-HIVEFORMER_version164906
  slide_cabinet_open_and_place_cups-HIVEFORMER_version165804
  stack_cups-HIVEFORMER_version164902
  stack_blocks-HIVEFORMER_version164905
)
tasks=(
  close_door
  take_usb_out_of_computer
  place_shape_in_shape_sorter
  sweep_to_dustpan
  take_plate_off_colored_dish_rack
  place_hanger_on_rack
  plug_charger_in_power_supply
  reach_and_drag
  setup_checkers
  tower3
  straighten_rope
  wipe_desk
  change_channel
  tv_on
  slide_cabinet_open_and_place_cups
  stack_cups
  stack_blocks
)

data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val
num_episodes=100
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
    --output_file /home/zhouxian/git/hiveformer_theo/eval_new.json
#    --variations {0..60}
done
