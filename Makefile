.PHONY: build
build:
	cd data && unzip training.zip && unzip test.zip

notebook:
	ipython notebook