exp=uncleaned_code_exp1

for ckpt in $(ls -d $exp/*); do
  python eval.py --checkpoint $ckpt/best.pth --name $exp --model develop
done
