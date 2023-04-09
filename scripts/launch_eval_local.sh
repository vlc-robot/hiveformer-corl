# HIVEFORMER
exp=03_24_hiveformer_setting
ckpts=(
  beat_the_buzz-HIVEFORMER_version161174
  close_grill-HIVEFORMER_version161176
  close_laptop_lid-HIVEFORMER_version161480
  hang_frame_on_hanger-HIVEFORMER_version161481
  open_door-HIVEFORMER_version161482
  open_window-HIVEFORMER_version161483
  put_umbrella_in_umbrella_stand-HIVEFORMER_version161652
  scoop_with_spatula-HIVEFORMER_version161653
  take_frame_off_hanger-HIVEFORMER_version161654
  take_toilet_roll_off_stand-HIVEFORMER_version161656
  close_box-HIVEFORMER_version161658
  insert_onto_square_peg-HIVEFORMER_version161659
  insert_usb_in_computer-HIVEFORMER_version161660
  move_hanger-HIVEFORMER_version161663
  open_oven-HIVEFORMER_version161968
  phone_on_base-HIVEFORMER_version161969
  place_shape_in_shape_sorter_version158810
  put_books_on_bookshelf-HIVEFORMER_version161973
  sweep_to_dustpan-HIVEFORMER_version161975
  take_plate_off_colored_dish_rack-HIVEFORMER_version161976
  water_plants-HIVEFORMER_version161977
  push_buttons-HIVEFORMER_version162804
  reach_and_drag-HIVEFORMER_version162805
  setup_checkers-HIVEFORMER_version162807
  tower3-HIVEFORMER_version162809
  screw_nail-HIVEFORMER_version162806
  wipe_desk-HIVEFORMER_version162810
  stack_cups-HIVEFORMER_version162502
  stack_blocks-HIVEFORMER_version162501
  slide_cabinet_open_and_place_cups-HIVEFORMER_version163033
  place_hanger_on_rack-HIVEFORMER_version163041
  plug_charger_in_power_supply-HIVEFORMER_version163042
)
tasks=(
  beat_the_buzz
  close_grill
  close_laptop_lid
  hang_frame_on_hanger
  open_door
  open_window
  put_umbrella_in_umbrella_stand
  scoop_with_spatula
  take_frame_off_hanger
  take_toilet_roll_off_stand
  close_box
  insert_onto_square_peg
  insert_usb_in_computer
  move_hanger
  open_oven
  phone_on_base
  place_shape_in_shape_sorter
  put_books_on_bookshelf
  sweep_to_dustpan
  take_plate_off_colored_dish_rack
  water_plants
  push_buttons
  reach_and_drag
  setup_checkers
  tower3
  screw_nail
  wipe_desk
  stack_cups
  stack_blocks
  slide_cabinet_open_and_place_cups
  place_hanger_on_rack
  plug_charger_in_power_supply
)

# PERACT
#exp=03_24_peract_setting
#ckpts=(
#)
#tasks=(
#)

data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val
num_episodes=100
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json
use_instruction=0
num_ghost_points=10000
headless=1
offline=0

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline $offline --num_episodes $num_episodes --headless $headless --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file --num_ghost_points $num_ghost_points --num_ghost_points_val $num_ghost_points
    # --output_file /home/zhouxian/git/hiveformer_theo/eval_new.json
    # --variations {0..60}
done


#exp=03_13_hiveformer_setting
#ckpts=(
#  slide_block_to_target-HIVEFORMER-SETTING_version157318
#  put_money_in_safe-HIVEFORMER-SETTING_version157317
#  take_money_out_safe-HIVEFORMER-SETTING_version157319
#  take_umbrella_out_of_umbrella_stand-HIVEFORMER-SETTING_version157320
#  pick_and_lift-HIVEFORMER-SETTING_version157314
#  pick_up_cup-HIVEFORMER-SETTING_version157315
#  put_knife_on_chopping_board-HIVEFORMER-SETTING_version157316
#)
#tasks=(
#  slide_block_to_target
#  put_money_in_safe
#  take_money_out_safe
#  take_umbrella_out_of_umbrella_stand
#  pick_and_lift
#  pick_up_cup
#  put_knife_on_chopping_board
#)
#data_dir=/home/theophile_gervet_gmail_com/datasets/raw/10_hiveformer_tasks_val
##data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
#num_episodes=50
#
#num_ckpts=${#ckpts[@]}
#for ((i=0; i<$num_ckpts; i++)); do
#  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#    --data_dir $data_dir --offline 0 --num_episodes $num_episodes \
#    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction 0
#done


#exp=02_20_compare_hiveformer_and_baseline
#ckpts=(
#  HIVEFORMER-put_money_in_safe_version153380
#  HIVEFORMER-slide_block_to_target_version153382
#  HIVEFORMER-take_umbrella_out_of_umbrella_stand_version153385
#)
#tasks=(
#  put_money_in_safe
#  slide_block_to_target
#  take_umbrella_out_of_umbrella_stand
#)
#data_dir=/home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1
#image_size="128,128"
#num_episodes=10
#
#num_ckpts=${#ckpts[@]}
#for ((i=0; i<$num_ckpts; i++)); do
#  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#    --data_dir $data_dir --image_size $image_size --offline 0 --num_episodes $num_episodes \
#    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE-HIVEFORMER --model original
#done
