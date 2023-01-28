exp=ghost_points_exp1

for ckpt in $(ls -d ghost_points_exp1/*); do
  python eval.py --checkpoint $ckpt/best.pth --name $exp
done
