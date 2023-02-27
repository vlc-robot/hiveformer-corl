## Data Preprocessing

```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/hiveformer/raw
output_dir=$root/datasets/hiveformer/packaged
train_seed=2
val_seed=3
train_episodes_per_task=100
val_episodes_per_task=100
task_file=tasks/2_debugging_tasks.csv

nohup sudo X &
export DISPLAY=:0.0
```

### 1 - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/$train_seed \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=256,256 \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=1
    
python dataset_generator.py \
    --save_path=$data_dir/$val_seed \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=256,256 \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=1
```

### 2 - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for seed in $train_seed $val_seed; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$seed \
            --output=$output_dir/$seed \
            --image_size=256,256 \
            --max_variations=1 \
            --tasks=$task
    done
done
```

### 3 - Preprocess instructions
```
cd $root/hiveformer
python -m data_preprocessing.preprocess_instructions \
    --tasks $(cat $task_file | tr '\n' ' ') \
    --output instructions.pkl \
    --variations {0..199} \
    --annotations data_preprocessing/annotations.json
```

### 4 - Compute workspace bounds
```
cd $root/hiveformer
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $root/datasets/hiveformer/packaged/0
```
