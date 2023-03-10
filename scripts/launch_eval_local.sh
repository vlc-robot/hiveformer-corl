exp=03_08_data_augmentations3
ckpts=(
  pick_and_lift-rescale-1.0,1.0-rotate-0.0_version156509
  pick_and_lift-rescale-1.0,1.0-rotate-45.0_version156510
  pick_up_cup-rescale-1.0,1.0-rotate-0.0_version156511
  pick_up_cup-rescale-1.0,1.0-rotate-45.0_version156512
)
tasks=(
  pick_and_lift
  pick_and_lift
  pick_up_cup
  pick_up_cup
)
data_dir=/home/theophile_gervet_gmail_com/datasets/raw/10_hiveformer_tasks_val
num_episodes=50

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline 0 --num_episodes $num_episodes --num_sampling_level 2 --regress_position_offset 1 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE
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
