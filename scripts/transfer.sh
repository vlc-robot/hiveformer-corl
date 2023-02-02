source_prefix=home/tgervet/hiveformer/xp
#source_prefix=home/tgervet/hiveformer/vln-robot/alhistory/xp
target_prefix=home/theophile_gervet_gmail_com/hiveformer
#exp=overfit
exp=overfit_cross_entropy
#exp=debug_mask2former_exp1
ckpt=best.pth
#ckpt=model.step=23976-value=0.00000.pth
#ckpt=checkpoints/epoch=0-step=50000.ckpt

# Get Tensorboard from source
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/*/events.out*" .
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/lightning_logs/*/events.out*" .

# Get checkpoints from source
sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/*/$ckpt" .
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/lightning_logs/*/$ckpt" .

# Send all to target
scp -r $source_prefix/$exp $GCLOUD:/$target_prefix/
