exp=reproduce_uncleaned_with_released_exp1/

for ckpt in $(ls -d $exp/*); do
  python eval.py --checkpoint $ckpt/best.pth --data_dir /home/theophile_gervet_gmail_com/datasets/hiveformer/raw/1 --name $exp --offline 0
done
