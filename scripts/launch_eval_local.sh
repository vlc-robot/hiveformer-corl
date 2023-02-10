exp=02_09_num_ghost_points
ckpts=(
  pick_up_cup-1000_version151770
)
tasks=(
  pick_up_cup
)

num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir /home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1 --offline 1 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-OFFLINE &
  python eval.py --tasks ${tasks[$i]} --checkpoint $exp/${ckpts[$i]}/best.pth \
    --data_dir /home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1 --offline 0 \
    --exp_log_dir $exp --run_log_dir ${tasks[$i]}-ONLINE &
done
