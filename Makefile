.PHONY: build
build:
	cd data && unzip training.zip && unzip test.zip

.PHONY: notebook
notebook:
	ipython notebook

.PHONY: test
test:
	nosetests --nocapture

.PHONY: run
run:
	python nn_connected.py

.PHONY: run_docker
run_docker:
	docker run -v /Users/guillaume/Documents/Development/Repositories/keypoints:/tf -it -p 8888:8888 -p 6006:6006 lungos/tensorflow /bin/sh -c 'cd /tf && python nn_connected.py'

.PHONY: start_docker
start_docker:
	docker run -v /Users/guillaume/Documents/Development/Repositories/keypoints:/tf -it -p 8888:8888 -p 6006:6006 lungos/tensorflow 

