## Conda Environment Setup

Only Linux is supported by RLBench.
```
conda create -n analogical_manipulation python=3.9
conda activate analogical_manipulation;
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch;
pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py matplotlib scipy tensorboard opencv-python open3d trimesh;
# Not needed anymore: dumped the submodules in this repo to make a few changes
#git submodule update --init --recursive

# Install PyRep
cd PyRep; 
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz; 
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
echo "export COPPELIASIM_ROOT=/home/theophile_gervet_gmail_com/hiveformer/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc; 
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc;
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc;
source ~/.bashrc;
pip install -r requirements.txt; pip install -e .; cd ..

# Install RLBench
cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;
sudo apt-get update; sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev;
sudo nvidia-xconfig -a --virtual=1280x1024;
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate;
sudo dpkg -i virtualgl*.deb; rm virtualgl*.deb;
sudo reboot  # Need to reboot for changes to take effect

# Install Mask2Former
git clone git@github.com:facebookresearch/Mask2Former.git
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git';
cd Mask2Former;
pip install -r requirements.txt;
cd mask2former/modeling/pixel_decoder/ops;
sh make.sh

# Install Mask3D (currently not used)
sudo apt install build-essential python3-dev libopenblas-dev
conda install openblas-devel -c anaconda
pip install ninja==1.10.2.3;
CUDA_HOME=/usr/local/cuda-11.3 pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas";
pip install torch-scatter -f https://data.pyg.org/whl/1.11.0+cu113.html;
pip install pytorch-lightning==1.7.2 fire imageio tqdm wandb python-dotenv pyviz3d scipy plyfile scikit-learn trimesh loguru albumentations volumentations;
pip install antlr4-python3-runtime==4.8;
pip install black==21.4b2;
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf';
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
cd third_party/pointnet2 && python setup.py install
```

Small changes to HiveFormer RLBench fork:
* add `set_callable_each_step` from [PerAct RLBench fork](https://github.com/MohitShridhar/RLBench/blob/peract/rlbench/action_modes/arm_action_modes.py)
* give `keyframe_actions` to `callable_each_step` in `get_demo()` of `rlbench/backend/scene.py`
```
# Record keyframe actions for visualization
keyframe_actions = np.stack([w._waypoint.get_matrix() for w in waypoints])
if callable_each_step is not None:
    callable_each_step(self.get_observation(), keyframe_actions=keyframe_actions)
```

## Dataset Generation

```
root=/home/theophile_gervet_gmail_com
data_dir=$root/datasets/hiveformer/raw
output_dir=$root/datasets/hiveformer/packaged
train_seed=0
val_seed=1
train_episodes_per_task=100
val_episodes_per_task=5
task_file=10_autolambda_tasks.csv

nohup sudo X &
export DISPLAY=:0.0

cd $root/hiveformer/RLBench/tools
python dataset_generator.py \
    --save_path=$data_dir/$train_seed \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=128,128 \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=1
python dataset_generator.py \
    --save_path=$data_dir/$val_seed \
    --tasks=$(cat $root/hiveformer/$task_file | tr '\n' ',') \
    --image_size=128,128 \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=1

cd $root/hiveformer
for task in $(cat $task_file | tr '\n' ' '); do
    for seed in $train_seed $val_seed; do
        python data_gen.py \
            --data_dir=$data_dir/$seed \
            --output=$output_dir/$seed \
            --max_variations=1 \
            --tasks=$task
    done
done

python preprocess_instructions.py \
    --tasks $(cat $task_file | tr '\n' ' ') \
    --output instructions.pkl \
    --variations {0..199} \
    --annotations annotations.json
```

## Training

Single task training:
```
#root=/home/theophile_gervet_gmail_com
#root=/opt
root=/home/tgervet
train_seed=0
val_seed=1
task_file=10_autolambda_tasks.csv
output_dir=$root/datasets/hiveformer/packaged
#DISPLAY=:0.0

# All tasks
for task in $(cat $task_file | tr '\n' ' '); do
    python train.py \
        --tasks $task \
        --dataset $output_dir/$train_seed \
        --valset $output_dir/$val_seed
done

# One specific task
python train.py \
    --tasks pick_and_lift \
    --dataset $output_dir/$train_seed \
    --valset $output_dir/$val_seed
```

## Evaluation

```
# One specific task
python eval.py \
    --tasks pick_and_lift \
    --checkpoint /path/to/checkpoint
```

## Docker

Install Nvidia Docker
```
sudo apt install -y nvidia-docker2
sudo systemctl daemon-reload
sudo systemctl restart docker
```
following [this](https://github.com/NVIDIA/nvidia-docker/issues/953) if run into an issue.

Build image and start container:
```
sudo DOCKER_BUILDKIT=1 docker build -f Dockerfile . -t hiveformer
sudo docker run --privileged --runtime=nvidia --gpus all -it -v /home/theophile_gervet_gmail_com/hiveformer:/opt/hiveformer -v /home/theophile_gervet_gmail_com/datasets/hiveformer:/opt/datasets/hiveformer -v /usr/bin/nvidia-xconfig:/usr/bin/nvidia-xconfig hiveformer:latest bash
```

## Issues Faced

[Resolved] CoppeliaSim requires XServer:
* `python dataset_generator.py [...]` fails because
* `cd $COPPELIASIM_ROOT ; ./coppeliaSim.sh` fails because
* `sudo X` fails
* https://github.com/stepjam/RLBench/issues/139 (problem)
* https://github.com/stepjam/RLBench/issues/142 (problem)
* https://github.com/Unity-Technologies/obstacle-tower-env/issues/51 (solution)

[Unresolved] Same issue within a Docker container:
* When using https://github.com/bengreenier/docker-xvfb, `cd $COPPELIASIM_ROOT ; ./coppeliaSim.sh` works fine but `python train.py` fails
* When trying to follow [these instructions](https://github.com/stepjam/RLBench/blob/master/README.md#running-headless) and start an X server with `X` as in the conda instructions within the container, we can't start the X server
* Same problem when trying to follow headless instructions from https://github.com/askforalfred/alfred
* https://github.com/stepjam/PyRep/issues/297
* https://github.com/stepjam/PyRep/issues/150
* https://github.com/NVIDIA/nvidia-docker/issues/529
* https://github.com/askforalfred/alfred/issues/61
* https://github.com/soyeonm/FILM/issues/3
* https://github.com/askforalfred/alfred/issues/112
* https://github.com/Bumblebee-Project/Bumblebee/issues/526

Need to install Nvidia drivers and CUDA toolkit on Google Cloud:
* [Drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
* [CUDA toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
