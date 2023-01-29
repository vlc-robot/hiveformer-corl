# We can override them with environment variables
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DATASET:=$(ROOT_DIR)/dataset

all: help download yarr rlbench pyrep install train
.PHONY: all

help :           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

download: 	## Download pre-computed dataset
	wget https://ucc0af9c27c68ea42b4bbf941b5b.dl.dropboxusercontent.com/zip_download_get/BFwM3u_b7IYpea9_x6HM4MJM3kYcp33dcE0c3Ktx-jcv8NK5d1DHlm_g5q65bVj0JvyabDdxK6uy3Ore-aoNuncryt55mVYF6CaxowdoEzgZig?_download_id=263523732926148580060100192660148252867197012399997912415770005 --no-check-certificate -O data.zip
	mkdir -p $(DATASET)
	unzip data.zip  -d $(DATASET)
	rm data.zip


yarr : 		## Install YARR dependency
	cd ../YARR && pip install -e . && cd ../auto-lambda

rlbench : 	## Install RLBench
	cd ../RLBench && pip install -r requirements.txt && pip install -e . && cd ../auto-lambda

pyrep : 	## Install PyRep
	cd ../PyRep && pip install -r requirements.txt && pip install -e . && cd ../auto-lambda

install: download yarr rlbench pyrep ## install dependencies
	# pip install poetry	
	# poetry install
	pip install -r requirements.txt

train: ## Run training
	# poetry shell
	python multi_task_baselines.py
