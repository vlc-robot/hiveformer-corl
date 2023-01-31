exp=ghost_points_exp2

for ckpt in $(ls -d $exp/*); do
  python eval.py --checkpoint $ckpt/best.pth --data_dir /home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1 --name $exp --model develop
done
