exp=uncleaned_code_02_15
ckpts=(
  version_152731/checkpoints/epoch=0-step=90000.ckpt
  version_152730/checkpoints/epoch=0-step=90000.ckpt
  version_152728/checkpoints/epoch=0-step=90000.ckpt
  version_152727/checkpoints/epoch=0-step=90000.ckpt
  version_152726/checkpoints/epoch=0-step=90000.ckpt
  version_152725/checkpoints/epoch=0-step=90000.ckpt
  version_152724/checkpoints/epoch=0-step=90000.ckpt
  version_152723/checkpoints/epoch=0-step=90000.ckpt
  version_152722/checkpoints/epoch=0-step=90000.ckpt
)
tasks=(
  take_umbrella_out_of_umbrella_stand
  take_money_out_safe
  slide_block_to_target
  reach_target
  push_button
  put_money_in_safe
  put_knife_on_chopping_board
  pick_up_cup
  pick_and_lift
)
num_ckpts=${#ckpts[@]}
for ((i=0; i<$num_ckpts; i++)); do
  python eval.py --tasks ${tasks[$i]} --checkpoint ../../$exp/lightning_logs/${ckpts[$i]}
done
