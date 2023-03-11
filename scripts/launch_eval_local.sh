exp=03_09_peract_setting
ckpts=(
  open_drawer-WITH-INSTRUCTION_version156726
  turn_tap-WITH-INSTRUCTION_version156730
  place_wine_at_rack_location-WITH-INSTRUCTION_version156737
  put_groceries_in_cupboard-WITH-INSTRUCTION_version156738
  insert_onto_square_peg-WITH-INSTRUCTION_version156741
)
tasks=(
  open_drawer
  turn_tap
  place_wine_at_rack_location
  put_groceries_in_cupboard
  insert_onto_square_peg
)
#data_dir=/home/theophile_gervet_gmail_com/datasets/raw/10_hiveformer_tasks_val
data_dir=/home/zhouxian/git/datasets/raw/18_peract_tasks_val
num_episodes=50

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline 0 --num_episodes $num_episodes --num_sampling_level 2 --regress_position_offset 1 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE --record_videos 0 --use_instruction 1
done


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
