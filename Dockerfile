# From https://github.com/bengreenier/docker-xvfb/
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
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
	python3.9-full \
	python3.9-dev \
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
  && rm -rf /var/lib/apt/lists/*

# Install coppelia
RUN mkdir -p /opt
WORKDIR /opt
RUN wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
	tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C /opt && \
	rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Do extra step from PyRep
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3.9 get-pip.py && \
	python3.9 -m pip --version
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
COPY PyRep /opt/PyRep/
WORKDIR /opt/PyRep
RUN python3.9 -m pip install -r requirements.txt && \
	python3.9 -m pip install -e . 

# And finally install Hiveformer dependencies
COPY requirements.txt /opt/requirements.txt
RUN python3.9 -m pip install -r /opt/requirements.txt

# Install Xvfb through external script
WORKDIR /usr/bin
COPY xvfb-startup.sh .
RUN sed -i 's/\r$//' xvfb-startup.sh
ARG RESOLUTION="224x224x24"
ENV XVFB_RES="${RESOLUTION}"
ARG XARGS=""
ENV XVFB_ARGS="${XARGS}"
ENTRYPOINT ["/bin/bash", "xvfb-startup.sh"]

# Test if everything is fine
CMD glxgears
