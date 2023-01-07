## Conda Environment Setup

```
conda create -n analogical_manipulation python=3.9
conda activate analogical_manipulation
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
pip install numpy pillow einops typed-argument-parser tqdm transformers
# TODO PyRep install is incomplete
cd PyRep && pip install -r requirements.txt && pip install -e . && cd ..
cd RLBench && pip install -r requirements.txt && pip install -e . && cd ..
```

## Dataset Generation

```
data_dir=/Users/theophile/Documents/Datasets/hiveformer/raw
output_dir=/Users/theophile/Documents/Datasets/hiveformer/packaged
seed=0

cd /Users/theophile/Documents/Projects/hiveformer/RLBench/tools
python dataset_generator.py \
--save_path=$data_dir/$seed \
--tasks=$(cat tasks.csv | tr '\n' ' ') \
--image_size=128,128 \
--renderer=opengl \
--episodes_per_task=100 \
--variations=1 \
--offset=0 \
--processes=1

cd /Users/theophile/Documents/Projects/hiveformer
for task in $(cat tasks.csv | tr '\n' ' '); do
    python data_gen.py \
    --data_dir=$data_dir/seed \
    --output=$output_dir \
    --max_variations=1 \
    --num_episodes=100 \
    --tasks=$task \
done
```
