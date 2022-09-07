all :
	container rlbench pyrep
.PHONY : all

help :           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

rlbench : 	## Install RLBench
	cd RLBench && pip install -r requirements.txt && pip install -e . && cd ..

pyrep : 	## Install PyRep
	cd PyRep && pip install -r requirements.txt && pip install -e . && cd ..

container:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile . -t hiveformer
	singularity build hiveformer.sif docker-daemon://hiveformer:latest
