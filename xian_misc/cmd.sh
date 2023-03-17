rsync -avzh  --info=progress2 --exclude-from='/home/zhouxian/git/hiveformer/rsync-exclude-file.txt' /home/zhouxian/git/hiveformer xianz1@matrix.ml.cmu.edu:/home/xianz1/git/
rsync -avzh --info=progress2  xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/02_27* home/zhouxian/git/hiveformer/train_logs/
rsync -avzh --info=progress2 --exclude-from='/home/xianz1/git/hiveformer/rsync-exclude-pth.txt' /home/xianz1/git/hiveformer/train_logs/02_28*  zhouxian@128.2.177.224:/home/zhouxian/git/hiveformer/train_logs/
rsync -avzh --info=progress2  --exclude-from='/home/zhouxian/git/hiveformer/xian_misc/rsync-exclude-pth.txt' xianz1@matrix.ml.cmu.edu:/home/xianz1/git/debug/hiveformer/train_logs/03_16*  /home/zhouxian/git/debug/hiveformer/train_logs/
rsync -avzh --info=progress2  xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/03_13/pick_and_lift-offset1-N3-T1000-V10000-symrot0-B16-lr1e-4_version157395/model.step=160000-value=0.00000.pth  /home/zhouxian/git/hiveformer/train_logs/03_13/pick_and_lift-offset1-N3-T1000-V10000-symrot0-B16-lr1e-4_version157395/
rsync -avzh --info=progress2  xianz1@matrix.ml.cmu.edu:/home/xianz1/git/hiveformer/train_logs/03_13/pick_and_lift-offset0-N3-T1000-V10000-symrot0-B16-lr1e-4_version157391/model.step=200000-value=0.00000.pth  /home/zhouxian/git/hiveformer/train_logs/03_13/pick_and_lift-offset0-N3-T1000-V10000-symrot0-B16-lr1e-4_version157391/
rsync -avzh --info=progress2 xianz1@matrix.ml.cmu.edu:/home/tgervet/datasets/hiveformer/packaged/3/put_money*  /home/zhouxian/git/datasets/hiveformer/packaged/3/
rsync -avzh --info=progress2 xianz1@matrix.ml.cmu.edu:/projects/katefgroup/analogical_manipulation/rlbench/raw/74_hiveformer_tasks_val/pick_and_lift*  /home/zhouxian/git/datasets/hiveformer/raw/74_hiveformer_tasks_val/
rsync -avzh --info=progress2 /home/xianz1/git/hiveformer/train_logs/02_27_rotation/put_money_in_safe-quat_from_kp-16-1e-4_version154556/model.step=75000-value=0.00757.pth  zhouxian@128.2.177.224:/home/zhouxian/git/hiveformer/train_logs/02_27_rotation/put_money_in_safe-quat_from_kp-16-1e-4_version154556/
rsync -avzh --info=progress2 /home/xianz1/git/hiveformer/train_logs/02_27_rotation/put_money_in_safe-quat_from_query-16-1e-4_version154557/model.step=95000-value=0.00814.pth  zhouxian@128.2.177.224:/home/zhouxian/git/hiveformer/train_logs/02_27_rotation/put_money_in_safe-quat_from_query-16-1e-4_version154557/

squeue -o "%u %c %m %b %P %M %N" |grep 2-25
squeue -o "%u %c %m %b %P %M %N" |grep 2-29
squeue -o "%u %c %m %b %P %M %N" |grep 1-22
squeue -o "%u %c %m %b %P %M %N" |grep 0-36
squeue -o "%u %c %m %b %P %M %N" |grep 1-24
squeue -o "%u %c %m %b %P %M %N" |grep 1-14
squeue -o "%u %c %m %b %P %M %N" |grep 0-16
squeue -o "%u %c %m %b %P %M %N" |grep 0-18
squeue -o "%u %c %m %b %P %M %N" |grep 0-22
# A100
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=62g  --nodelist=matrix-2-25 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=62g  --nodelist=matrix-2-29 --pty $SHELL 
# 2080ti
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-1-22 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=128g  --nodelist=matrix-1-22 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-36 --pty $SHELL 
# V100
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c20 --mem=100g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=240g  --nodelist=matrix-1-24 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:2 -c20 --mem=128g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:3 -c30 --mem=180g  --nodelist=matrix-1-14 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:4 -c40 --mem=240g  --nodelist=matrix-1-14 --pty $SHELL 
# Titan X
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=64g  --nodelist=matrix-0-16 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-18 --pty $SHELL 
srun -p kate_reserved --time=72:00:00  --gres gpu:1 -c10 --mem=32g  --nodelist=matrix-0-22 --pty $SHELL 