exp=03_13_peract_setting
ckpts=(
#  sweep_to_dustpan_of_size_version157137
#  push_buttons_version157147
#  slide_block_to_color_target_version157136
#  meat_off_grill_version157138
#  turn_tap_version157139
  close_jar_version157140
  reach_and_drag_version157141
  light_bulb_in_version157142
  put_money_in_safe_version157143
#  place_wine_at_rack_location_version157144
  put_groceries_in_cupboard_version157145
  place_shape_in_shape_sorter_version157146
  insert_onto_square_peg_version157148
)
tasks=(
#  sweep_to_dustpan_of_size
#  push_buttons
#  slide_block_to_color_target
#  meat_off_grill
#  turn_tap
  close_jar
  reach_and_drag
  light_bulb_in
  put_money_in_safe
#  place_wine_at_rack_location
  put_groceries_in_cupboard
  place_shape_in_shape_sorter
  insert_onto_square_peg
)
#data_dir=/home/theophile_gervet_gmail_com/datasets/raw/10_hiveformer_tasks_val
data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=10

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline 0 --num_episodes $num_episodes --headless 0 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction 1 --variations {0..60}
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
