source_prefix=home/tgervet/hiveformer/xp
#source_prefix=home/tgervet/hiveformer/vln-robot/alhistory/xp
target_prefix=home/theophile_gervet_gmail_com/hiveformer
#exp=overfit
exp_src=02_07/*
exp_tgt=02_07
#exp=debug_mask2former_exp1
ckpt=best.pth
#ckpt=model.step=23976-value=0.00000.pth
#ckpt=checkpoints/epoch=0-step=50000.ckpt

# Get Tensorboard from source
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp_src/*/events.out*" .
#sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/lightning_logs/*/events.out*" .

# Get checkpoints from source
sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp_src/*/$ckpt" .
#sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/lightning_logs/*/$ckpt" .

# Send all to target
scp -r $source_prefix/$exp_tgt $GCLOUD:/$target_prefix/
