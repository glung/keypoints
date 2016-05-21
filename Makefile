.PHONY: build
build:
	cd data && unzip training.zip && unzip test.zip

.PHONY: notebook
notebook:
	ipython notebook

.PHONY: test
test:
	nosetests --nocapture
