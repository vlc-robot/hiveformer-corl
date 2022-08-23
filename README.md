# :bee: Hiveformer :bee:

[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


This is the PyTorch implementation of the Hiveformer research paper.



## :hammer_and_wrench: 1. Getting started

You need a recent version of Python (higher or equal than 3.9) and install dependencies:

```bash
poetry install

# this should be run on every shell
poetry shell
```

Other dependencies need to be installed manually. See details on [RLBench webpage](https://github.com/stepjam/RLBench).


## :minidisc: 2. Preparing dataset

Hiveformer is training over an offline dataset. You need thus to generate demonstrations:

```bash
cd /path/to/rlbench
data_dir=/path/to/dataset
for seed in 0 1 2 3 4; do
python dataset_generator.py \
  --save_path=$data_dir/$seed \
  --tasks=$(cat tasks.csv | tr '\n' ' ') \
  --image_size=128,128 \
  --renderer=opengl \
  --episodes_per_task=100 \
  --variations=1 \
  --offset=0 \
  --processes=1
  done
```

Then, you need to package demonstrations into small and fast-to-load numpy files:

```bash
cd /path/to/hiveformer
data_dir=/path/to/raw/dataset
output_dir=/path/to/packaged/dataset
for seed in 0 1 2 3 4 5; do
for task in $(cat tasks.csv | tr '\n' ' '); do
python data_gen.py \
    --data_dir=$data_dir/$seed \
    --output=$output_dir \
    --max_variations=1 \
    --num_episodes=100 \
    --tasks=$task \
done
done
```

Next, you are going to preprocess instructions:


```zsh
python preprocess_instructions.py \
	--tasks $(cat tasks.csv | tr '\n' ' ')
	--output instructions.pkl \
	--variations {0..199} \
	--annotations annotations.json
```


## :weight_lifting: 3. Train your agents

### 3.1. Single-task learning

```bash
for seed in 0 1 2 3 4 5; do
for task in $(cat tasks.csv | tr '\n' ' '); do
python train.py \
	--arch mct \
	--tasks $task \
	--dataset $output_dir/$seed \
	--num_workers 10  \
 	--instructions instructions.pkl \
	--variations 0
done
done
```

### 3.2. Multi-task learning

```bash
for seed in 0 1 2 3 4 5; do
python train.py \
	--arch mct \
	--tasks $(cat tasks.csv | tr '\n' ' ') \
	--dataset $output_dir/$seed \
	--num_workers 10  \
 	--instructions instructions.pkl \
	--variations 0
done
```

### 3.3. Multi-variation learning

```bash
for seed in 0 1 2 3 4 5; do
for task in push_buttons tower3; do
python train.py \
	--arch mct \
	--tasks $task \
	--dataset $output_dir/$seed \
	--num_workers 10  \
 	--instructions instructions.pkl \
	--variations {0..99}
done
done
```


## 4. :stopwatch: Evaluation

```
python eval.py \
	--checkpoint /path/to/checkpoint/ 
	--variations 0 \
	--instructions instructions.pkl \
	--num_episodes 100
```


## :pray: Credits

Parts of the code were copied from Auto-Lambda, Cosy Pose, LXMERT and RLBench.
