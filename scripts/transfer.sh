source_prefix=home/tgervet/hiveformer/xp
target_prefix=home/theophile_gervet_gmail_com/hiveformer
exp=ghost_points_exp1

# Get Tensorboard from source
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/*/events.out*" .

# Get checkpoints from source
sshpass -p $MATRIX_PW rsync -RL "$MATRIX:/$source_prefix/$exp/*/best.pth" .

# Send all to target
scp -r $source_prefix/$exp $GCLOUD:/$target_prefix/
