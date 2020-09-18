SHELL=/usr/bin/env bash

GPU_PROGRAM           ?= nvidia-smi
GPU_RESULT             = $(shell which $(GPU_PROGRAM) 2>/dev/null)
GPU_TEST               = $(notdir $(GPU_RESULT))

ifeq ($(GPU_TEST), $(GPU_PROGRAM))
	GPU_FLAG           = --gpus all
endif

PYTORCH_DOWNLOAD_LINK ?= https://download.pytorch.org/whl/test/cu101/torch_test.html
# Used by CI to do plain buildkit progress
BUILD_PROGRESS        ?=
DOCKER_BUILD           = cat Dockerfile | DOCKER_BUILDKIT=1 docker build --target $@ $(BUILD_PROGRESS) --build-arg "PYTORCH_DOWNLOAD_LINK=$(PYTORCH_DOWNLOAD_LINK)" -t pytorch/integration-testing:$@ -
DOCKER_RUN             = set -o pipefail; docker run --rm -it $(GPU_FLAG) --shm-size 8G -v "$(PWD)/output:/output" pytorch/integration-testing:$@
CHOWN_TO_USER          = docker run --rm -v "$(PWD)":/v -w /v alpine chown -R "$(shell id -u):$(shell id -g)" .


.PHONY: all
all:
	@echo "please specify your target"

logs/:
	mkdir -p logs/

.PHONY: pyro
pyro: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest -v -c /dev/null -n auto --color=no --junitxml=/output/$@_results.xml --stage unit | tee logs/$@.log

.PHONY: detectron2
detectron2: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	# We have to install detectron2 here since it won't actually do the cuda build unless it has direct access to the GPU, :shrug:
	$(DOCKER_RUN) \
		sh -c 'pip install -U -e /detectron2 && pytest -v --color=no --junitxml=/output/$@_results.xml /detectron2/tests' 2>&1 \
			| tee logs/$@.log

.PHONY: transformers
transformers: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest -n auto --dist=loadfile --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>&1 | tee logs/$@.log

.PHONY: fairseq
fairseq: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>/dev/null | tee logs/$@.log

.PHONY: chown-to-user
chown-to-user:
	$(CHOWN_TO_USER)

.PHONY: clean
clean: chown-to-user
	$(RM) -r output/
	$(RM) -r logs/
