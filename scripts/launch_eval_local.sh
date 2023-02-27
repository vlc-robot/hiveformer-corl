exp=02_26_MATCH_HIVEFORMER
ckpts=(
  FULL-pick_and_lift-BALL-0.16_version154273
  FULL-pick_up_cup-BALL-0.16_version154274
  FULL-put_knife_on_chopping_board-BALL-0.16_version154275
  FULL-put_money_in_safe-BALL-0.16_version154276
  FULL-slide_block_to_target-BALL-0.16_version154277
  FULL-take_money_out_safe-BALL-0.08_version154285
  FULL-take_umbrella_out_of_umbrella_stand-BALL-0.16_version154279
)
tasks=(
  pick_and_lift
  pick_up_cup
  put_knife_on_chopping_board
  put_money_in_safe
  slide_block_to_target
  take_money_out_safe
  take_umbrella_out_of_umbrella_stand
)
data_dir=/home/theophile_gervet_gmail_com/datasets/hiveformer/raw/3
image_size="256,256"
num_episodes=10

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --image_size $image_size --offline 0 --num_episodes $num_episodes \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE
#  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
#    --data_dir $data_dir --image_size $image_size --offline 1 --num_episodes $num_episodes \
#    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-OFFLINE
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
