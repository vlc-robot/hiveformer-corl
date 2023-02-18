exp=02_17_train_rotation2
ckpts=(
  put_money_in_safe-1000-quat_from_query_version153036
)
tasks=(
  put_money_in_safe
)
data_dir=/home/theophile_gervet_gmail_com/datasets/hiveformer/raw/3
image_size="256,256"
num_episodes=10

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --image_size $image_size --offline 1 --num_episodes $num_episodes \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-OFFLINE &
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --image_size $image_size --offline 0 --num_episodes $num_episodes \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE &
done
