source_prefix=home/tgervet/hiveformer/xp
target_prefix=home/theophile_gervet_gmail_com/hiveformer
exp=exp7

# Tensorboard
sshpass -p $MATRIX_PW rsync -R "$MATRIX:/$source_prefix/$exp/*/events.out*" .
scp -r $source_prefix/$exp $GCLOUD:/$target_prefix/