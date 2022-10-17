# :bee: Hiveformer :bee:

[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![arXiv](https://img.shields.io/badge/cs.RO-2209.04899-red.svg?style=for-the-badge&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2209.04899)
![Star this repository](https://img.shields.io/github/stars/guhur/hiveformer?style=for-the-badge)
[![site](https://img.shields.io/badge/🌐%20site-github%20pages-blue.svg?style=for-the-badge)](https://guhur.github.io/hiveformer)



This is the PyTorch implementation of the Hiveformer research paper:

> Instruction-driven history-aware policies for robotic manipulations  
> Pierre-Louis Guhur, Shizhe Chen, Ricardo Garcia, Makarand Tapaswi, Ivan Laptev, Cordelia Schmid  
> **CoRL 2022 (oral)**



## :hammer_and_wrench: 1. Getting started

Clone the repository along with its submodules:

```bash
git clone --recursive https://github.com/guhur/hiveformer

# or
git clone https://github.com/guhur/hiveformer
git submodule --init --recursive update
```

You need a recent version of Python (higher or equal than 3.9) and install dependencies:

```bash
poetry install

# this should be run on every shell
poetry shell
```

Other dependencies ([RLBench](https://github.com/stepjam/RLBench), [PyRep](https://github.com/stepjam/PyRep)) need to be installed manually. Launch corresponding Makefile rules:

```bash
make pyrep
make rlbench
```

Note that you can also use a Docker or Singularity container:

```bash
make container
```

## :minidisc: 2. Preparing dataset

Hiveformer is training over an offline dataset of succesful demonstrations. You can generate demonstrations from the corresponding [SLURM file](./slurm/generate-samples.slurm) or with the following command:

```bash
data_dir=/path/to/raw/dataset
output_dir=/path/to/packaged/dataset
for seed in 0 1 2 3 4; do
  cd /path/to/rlbench
  # Generate samples
  python dataset_generator.py \
    --save_path=$data_dir/$seed \
    --tasks=$(cat tasks.csv | tr '\n' ' ') \
    --image_size=128,128 \
    --renderer=opengl \
    --episodes_per_task=100 \
    --variations=1 \
    --offset=0 \
    --processes=1

  cd /path/to/hiveformer
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

Next, you need to preprocess instructions:

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
for seed in 0 1 2 3 4; do
python train.py \
	--tasks $(cat tasks.csv | tr '\n' ' ') \
	--dataset $output_dir/$seed \
	--num_workers 10  \
 	--instructions instructions.pkl \
	--variations 0
done
```

### 3.3. Multi-variation learning

```bash
for seed in 0 1 2 3 4; do
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
