mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
#IMG_TENSOR_FLOW := lungos/tensorflow
#IMG_TENSOR_FLOW := gcr.io/tensorflow/tensorflow:latest-devel
IMG_TENSOR_FLOW := keypoints/tensorflow

.PHONY: build
build:
	cd data && unzip training.zip && unzip test.zip

.PHONY: notebook
notebook:
	ipython notebook

.PHONY: test
test:
	nosetests --nocapture

.PHONY: run_sk
run_sk:
	python keypoints.py --model SKLEARN_LINEAR_RIGE

.PHONY: run_tf
run_tf:
	python tf.py

.PHONY: build_container
build_image:
	docker build . -t $(IMG_TENSOR_FLOW)

.PHONY: start_container
start_container: 
	docker run -v $(current_dir):/tf -v /tmp:/tmp -w /tf -it -p 8888:8888 -p 6006:6006 -e "PASSWORD=" $(IMG_TENSOR_FLOW)

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir target/tflearn_logs/ --port 6007
