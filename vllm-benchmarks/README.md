### Prerequisite

Prepare your HuggingFace token and save it into `HF_TOKEN` environment
variable. Note that the token needs to accept the terms and conditions
of all the test models in
[vLLM](https://github.com/vllm-project/vllm/tree/main/.buildkite/nightly-benchmarks/tests),
otherwise, the model will be skipped.

It's recommended to have ccache or sccache setup as building vLLM could
take sometimes.

### vLLM benchmark on PyTorch infra

* Run the benchmark on the latest commit in a branch, i.e. `main`

```
./run.sh main
```

* Run the benchmark on a specific commit on [vLLM](https://github.com/vllm-project/vllm)

```
./run.sh <COMMIT_SHA>
```

* Run the benchmark, but don't upload the results to PyTorch OSS
  benchmark database

```
UPLOAD_BENCHMARK_RESULTS=0 ./run.sh main
```
