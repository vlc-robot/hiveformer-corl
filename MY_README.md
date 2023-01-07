## Conda Environment Setup

Only Linux is supported by RLBench.
```
conda create -n analogical_manipulation python=3.9
conda activate analogical_manipulation;
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch;
pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py;
git submodule update --init --recursive

# Install PyRep
cd PyRep; 
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz; 
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
echo "export COPPELIASIM_ROOT=/home/theophile_gervet/Documents/hiveformer/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc; 
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc;
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc;
source ~/.bashrc;
pip install -r requirements.txt; pip install -e .; cd ..

# Install RLBench
cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;

# Possibly needed
sudo apt-get update; sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev;
sudo nvidia-xconfig -a --virtual=1280x1024;
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb --no-check-certificate;
sudo dpkg -i virtualgl*.deb; rm virtualgl*.deb;
sudo reboot
```

## Dataset Generation

```
root=/home/theophile_gervet
data_dir=$root/datasets/hiveformer/raw
output_dir=$root/datasets/hiveformer/packaged
seed=0

cd $root/hiveformer/RLBench/tools
nohup sudo X &
export DISPLAY=:0.0
python dataset_generator.py \
--save_path=$data_dir/$seed \
--tasks=$(cat $root/hiveformer/10_autolambda_tasks.csv | tr '\n' ' ') \
--image_size=128,128 \
--renderer=opengl \
--episodes_per_task=100 \
--variations=1 \
--offset=0 \
--processes=1

cd $root/hiveformer
for task in $(cat 10_autolambda_tasks.csv | tr '\n' ' '); do
    python data_gen.py \
    --data_dir=$data_dir/seed \
    --output=$output_dir \
    --max_variations=1 \
    --num_episodes=100 \
    --tasks=$task \
done
```

## Issues Faced

Issue 1:
* `python dataset_generator.py [...]` fails because
* `cd $COPPELIASIM_ROOT ; ./coppeliaSim.sh` fails because
* `sudo X` fails
* https://github.com/stepjam/RLBench/issues/139
* https://github.com/stepjam/RLBench/issues/142
* https://github.com/Unity-Technologies/obstacle-tower-env/issues/51
