exp=02_14_tune_ghost_points
ckpts=(
  put_money_in_safe-cube-0.08-ghost-2000-gtsample-0_version152530
)
tasks=(
  put_money_in_safe
)
data_dir=/home/theophile_gervet_gmail_com/datasets/hiveformer/raw/3
image_size="256,256"

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir $data_dir --offline 1 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-OFFLINE &
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --
    --data_dir $data_dir --offline 0 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE &
done
