FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN  apt-get update -y \
  && apt-get install --no-install-recommends -y \
  	xvfb \
	wget \
    tar \
	xz-utils \
    libx11-6 \
	libxcb1 \
	libxau6 \
    dbus-x11 \
	x11-utils \
	libxkbcommon-x11-0 \
    libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	python3-opengl \
	build-essential \
	libffi-dev \
	freeglut3-dev \
	freeglut3 \
	libglu1-mesa \
	libglu1-mesa-dev \
 	libgl1-mesa-glx \
	libgl1-mesa-dev \
	libgl1-mesa-dri \
	libxext-dev \
	libxt-dev \
	git \
    xorg \
    libxcb-randr0-dev \
    libxrender-dev \
    libxkbcommon-dev \
    mesa-utils \
    pciutils \
  && rm -rf /var/lib/apt/lists/*

# Copy submodules
COPY PyRep /opt/PyRep/
COPY RLBench /opt/RLBench/

# Install conda
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate  && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n analogical_manipulation python=3.9 -y

# Install Coppelia
RUN mkdir -p /opt
WORKDIR /opt
RUN wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz --no-check-certificate && \
	tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C /opt && \
	rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Install PyRep
WORKDIR /opt/PyRep
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate analogical_manipulation && \
    pip install -r requirements.txt && pip install -e .

# Install RLBench
WORKDIR /opt/RLBench/
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate analogical_manipulation && \
    pip install -r requirements.txt && pip install -e .

# Install Hiveformer dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate analogical_manipulation && \
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch && \
    pip install numpy pillow einops typed-argument-parser tqdm transformers absl-py matplotlib scipy tensorboard opencv-python open3d trimesh

# Install Xvfb with instructions from HiveFormer
#WORKDIR /usr/bin
#COPY xvfb-startup.sh .
#RUN sed -i 's/\r$//' xvfb-startup.sh
#ARG RESOLUTION="224x224x24"
#ENV XVFB_RES="${RESOLUTION}"
#ARG XARGS=""
#ENV XVFB_ARGS="${XARGS}"
#ENTRYPOINT ["/bin/bash", "xvfb-startup.sh"]

# Test if everything is fine
#CMD glxgears
