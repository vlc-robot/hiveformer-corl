# Data Generation

## 1 - HiveFormer Data Generation
```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/hiveformer/raw
output_dir=$root/datasets/hiveformer/packaged
train_dir=74_hiveformer_tasks_train
val_dir=74_hiveformer_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/74_hiveformer_tasks.csv

nohup sudo X &
export DISPLAY=:0.0
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=5

python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=5
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=1 \
            --tasks=$task
    done
done
```

### C - Compute workspace bounds
```
cd $root/hiveformer
task_file=tasks/74_hiveformer_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_root \
    --out_file 74_hiveformer_tasks_location_bounds.json
```

## 1 - PerAct Data Generation
```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/peract/raw
output_dir=$root/datasets/peract/packaged
train_dir=18_peract_tasks_train
val_dir=18_peract_tasks_val
train_episodes_per_task=100
val_episodes_per_task=100
image_size="256,256"
task_file=tasks/18_peract_tasks.csv
```

### A - Generate raw train and val data
```
cd $root/hiveformer/RLBench/tools

python dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=5
    
python dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=-1 \
    --offset=0 \
    --processes=5
```

### B - Preprocess train and val data
```
cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for split_dir in $train_dir $val_dir; do
        python -m data_preprocessing.data_gen \
            --data_dir=$data_dir/$split_dir \
            --output=$output_dir/$split_dir \
            --image_size=$image_size \
            --max_variations=60 \
            --tasks=$task
    done
done
```

### C - Compute workspace bounds
```
cd $root/hiveformer
task_file=tasks/18_peract_tasks.csv
python -m data_preprocessing.compute_workspace_bounds \
    --dataset $output_dir/$train_root \
    --out_file 18_peract_tasks_location_bounds.json
```

## 3 - Preprocess Instructions for Both Datasets
```
root=/home/theophile_gervet_gmail_com
cd $root/hiveformer
task_file=tasks/82_all_tasks.csv
python -m data_preprocessing.preprocess_instructions \
    --tasks $(cat $task_file | tr '\n' ' ') \
    --output instructions.pkl \
    --variations {0..199} \
    --annotations data_preprocessing/annotations.json
```
