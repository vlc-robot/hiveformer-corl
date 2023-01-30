#source_prefix=home/tgervet/hiveformer/xp
source_prefix=home/tgervet/hiveformer/vln-robot/alhistory/xp
target_prefix=home/theophile_gervet_gmail_com/hiveformer
exp=uncleaned_code_exp2
#ckpt=best.pth
ckpt=checkpoints/epoch=0-step=70000.ckpt

# Get Tensorboard from source
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/*/events.out*" .
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/lightning_logs/*/events.out*" .

# Get checkpoints from source
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/*/$ckpt" .
sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/lightning_logs/*/$ckpt" .

# Send all to target
scp -r $source_prefix/$exp $GCLOUD:/$target_prefix/
