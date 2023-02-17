# Ours

source_prefix=home/tgervet/hiveformer/train_logs
target_prefix=home/theophile_gervet_gmail_com/hiveformer
exp_src=02_17_train_with_offset2
exp_tgt=02_17_train_with_offset2
#exp_src=02_17_overfit_rotation2
#exp_tgt=02_17_overfit_rotation2
ckpt=best.pth

# Get Tensorboard from source
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/*/events.out*" .

# Get checkpoints from source
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/*/$ckpt" .

# Send all to target
scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/


# Uncleaned HiveFormer

#source_prefix=home/tgervet/hiveformer/train_logs
#source_prefix=home/tgervet/hiveformer/vln-robot/alhistory/xp
#exp_src=uncleaned_code_02_15
#exp_tgt=uncleaned_code_02_15
#ckpt=checkpoints/epoch=0-step=90000.ckpt
#
## Get Tensorboard from source
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/lightning_logs/*/events.out*" .
#
## Get checkpoints from source
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/lightning_logs/*/$ckpt" .
#
## Send all to target
#scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/