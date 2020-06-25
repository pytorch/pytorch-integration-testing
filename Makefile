PYTORCH_DOWNLOAD_LINK ?= https://download.pytorch.org/whl/test/cu101/torch_test.html
DOCKER_BUILD           = cat Dockerfile | docker build --target $@ --build-arg "PYTORCH_DOWNLOAD_LINK=$(PYTORCH_DOWNLOAD_LINK)" -t pytorch/integration-testing:$@ -
DOCKER_RUN             = docker run --rm -it --gpus all --shm-size 8G -v "$(PWD)/output:/output" pytorch/integration-testing:$@
CHOWN_TO_USER          = docker run --rm -v "$(PWD)":/v -w /v alpine chown -R "$(shell id -u):$(shell id -g)" .


.PHONY: all
all:
	@echo "please specify your target"

logs/:
	mkdir -p logs/

.PHONY: fastai
fastai: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) py.test -v --color=no --junitxml=/output/$@_results.xml tests | tee logs/$@.log
	$(CHOWN_TO_USER)

.PHONY: pyro
pyro: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pytest -v -c /dev/null -n auto --color=no --junitxml=/output/$@_results.xml --stage unit | tee logs/$@.log
	$(CHOWN_TO_USER)

.PHONY: detectron2
detectron2: logs/
	$(DOCKER_BUILD)
	# We have to install detectron2 here since it won't actually do the cuda build unless it has direct access to the GPU, :shrug:
	$(DOCKER_RUN) \
		sh -c 'pip install -U -e /detectron2 && pytest -v --color=no --junitxml=/output/$@_results.xml /detectron2/tests' 2>/dev/null \
			| tee logs/$@.log

.PHONY: transformers
transformers: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pytest -n auto --dist=loadfile --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>/dev/null | tee logs/$@.log

.PHONY: fairseq
fairseq: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pytest --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>/dev/null | tee logs/$@.log

.PHONY: clean
clean:
	$(RM) -r output/
	$(RM) -r logs/
