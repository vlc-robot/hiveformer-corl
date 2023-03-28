exp=03_24_hiveformer_setting
ckpts=(
  turn_tap-HIVEFORMER_version160617
  push_button-HIVEFORMER_version160614
  slide_block_to_target-HIVEFORMER_version160615
  press_switch-HIVEFORMER_version160613
  take_usb_out_of_computer-HIVEFORMER_version160616
  lamp_off-HIVEFORMER_version160612
  close_drawer-HIVEFORMER_version160609
  close_fridge-HIVEFORMER_version160610
  close_microwave-HIVEFORMER_version160611
  reach_target-HIVEFORMER_version160608
  close_grill-HIVEFORMER_version161176
  beat_the_buzz-HIVEFORMER_version161174
  change_clock-HIVEFORMER_version161175
  basketball_in_hoop-HIVEFORMER_version161173
  pick_up_cup-HIVEFORMER_version161167
  play_jenga-HIVEFORMER_version161168
  take_lid_off_saucepan-HIVEFORMER_version161169
  toilet_seat_up-HIVEFORMER_version161171
  take_umbrella_out_of_umbrella_stand-HIVEFORMER_version161170
  turn_oven_on-HIVEFORMER_version161172
  open_wine_bottle-HIVEFORMER_version161075
  open_microwave-HIVEFORMER_version161074
  open_grill-HIVEFORMER_version161073
  open_drawer-HIVEFORMER_version161071
  open_fridge-HIVEFORMER_version161072
  lift_numbered_block-HIVEFORMER_version161069
  open_box-HIVEFORMER_version161070
  close_door-HIVEFORMER_version161046
  unplug_charger-HIVEFORMER_version161045
  lamp_on-HIVEFORMER_version161047
)
tasks=(
  turn_tap
  push_button
  slide_block_to_target
  press_switch
  take_usb_out_of_computer
  lamp_off
  close_drawer
  close_fridge
  close_microwave
  reach_target
  close_grill
  beat_the_buzz
  change_clock
  basketball_in_hoop
  pick_up_cup
  play_jenga
  take_lid_off_saucepan
  toilet_seat_up
  take_umbrella_out_of_umbrella_stand
  turn_oven_on
  open_wine_bottle
  open_microwave
  open_grill
  open_drawer
  open_fridge
  lift_numbered_block
  open_box
  close_door
  unplug_charger
  lamp_on
)
data_dir=/home/zhouxian/git/datasets/raw/74_hiveformer_tasks_val
num_episodes=2
gripper_loc_bounds_file=tasks/74_hiveformer_tasks_location_bounds.json

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline 0 --num_episodes $num_episodes --headless 0 --output_file eval/${tasks[$i]}.json  \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction $use_instruction \
    --gripper_loc_bounds_file $gripper_loc_bounds_file
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
