SHELL=/usr/bin/env bash

BASE_IMAGE      ?= ubuntu:14.04
BASE            ?= ubuntu-base
PYTORCH_REF     ?= master
PYTORCH_CHANNEL ?= pytorch-test
BUILD_PROGRESS  ?=

# We designate IN_CIRCLECI=1 to have the runner generate xml reports
DOCKER_BUILD = docker build \
	-t compat-test:$@ \
	$(BUILD_PROGRESS) \
	--build-arg BASE_IMAGE=$(BASE_IMAGE) \
	--build-arg BASE=$(BASE) \
	--build-arg PYTORCH_REF=$(PYTORCH_REF) \
	--build-arg PYTORCH_CHANNEL=$(PYTORCH_CHANNEL) \
	.
DOCKER_RUN   = set -o pipefail; docker run --rm -it -e IN_CIRCLECI=1  -w /pytorch/test -v "$(PWD)/test-reports:/pytorch/test/test-reports" compat-test:$@

.PHONY: ubuntu-16.04
ubuntu-16.04: BASE_IMAGE := ubuntu:16.04
ubuntu-16.04: BASE       := debian-base
ubuntu-16.04:
	$(DOCKER_BUILD)
	$(DOCKER_RUN) | tee $@.log

.PHONY: ubuntu-18.04
ubuntu-18.04: BASE_IMAGE := ubuntu:18.04
ubuntu-18.04: BASE       := debian-base
ubuntu-18.04:
	$(DOCKER_BUILD)
	$(DOCKER_RUN) | tee $@.log

.PHONY: centos-7
centos-7: BASE_IMAGE := centos:7
centos-7: BASE       := redhat-base
centos-7:
	$(DOCKER_BUILD)
	$(DOCKER_RUN) | tee $@.log

.PHONY: archlinux
archlinux: BASE_IMAGE := archlinux
archlinux: BASE       := arch-base
archlinux:
	$(DOCKER_BUILD)
	$(DOCKER_RUN) | tee $@.log

.PHONY: macos-compat
macos-compat:
	cd ../pytorch/test && ../../compat-tests/entrypoint.sh
