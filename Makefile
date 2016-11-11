mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
IMG_TENSOR_FLOW := lungos/tensorflow

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

.PHONY: start_container
start_container:
	docker run -v $(current_dir):/tf -w /tf -it -p 8888:8888 -p 6006:6006 $(IMG_TENSOR_FLOW)
