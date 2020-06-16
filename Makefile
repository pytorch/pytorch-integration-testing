DOCKER_BUILD=cat Dockerfile | docker build --target $@ -t pytorch/integration-testing:$@ -
DOCKER_RUN=docker run --rm --gpus all -it -v "$(PWD)/output:/output" pytorch/integration-testing:$@
CHOWN_TO_USER=docker run --rm -v "$(PWD)":/v -w /v alpine chown -R "$(shell id -u):$(shell id -g)" .

.PHONY: all
all:
	@echo "please specify your target"

.PHONY: fastai
fastai:
	$(DOCKER_BUILD)
	mkdir -p logs/
	$(DOCKER_RUN) py.test -v --color=no --junitxml=/output/$@_results.xml tests | tee logs/$@.log
	$(CHOWN_TO_USER)

.PHONY: pyro
pyro:
	$(DOCKER_BUILD)
	mkdir -p logs/
	$(DOCKER_RUN) pytest -v -c /dev/null -n auto --color=no --junitxml=/output/$@_results.xml --stage unit | tee logs/$@.log
	$(CHOWN_TO_USER)

.PHONY: clean
clean:
	$(RM) -r output/
	$(RM) -r logs/
