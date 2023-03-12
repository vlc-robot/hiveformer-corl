# Conda Environment Setup

We train on Matrix without RLbench, and evaluate locally with RLBench.

Environment setup on both Matrix and locally:
```
conda create -n analogical_manipulation python=3.9
conda activate analogical_manipulation;
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch;
pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py matplotlib scipy tensorboard opencv-python open3d trimesh wandb;
pip install git+https://github.com/openai/CLIP.git;

# PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath;
conda install pytorch3d -c pytorch3d;
```

To install RLBench locally:
```
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
```

## Dataset Generation

See `data_preprocessing` folder.

## Training

See `scripts/launch_train_slurm.sh`.

## Evaluation

See `scripts/launch_eval_local.sh`.
