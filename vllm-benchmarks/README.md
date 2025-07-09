### Prerequisite

Prepare your HuggingFace token and save it into `HF_TOKEN` environment
variable. Note that the token needs to accept the terms and conditions
of all the test models in
[vLLM](https://github.com/vllm-project/vllm/tree/main/.buildkite/nightly-benchmarks/tests),
otherwise, the model will be skipped.
TEST
It's recommended to have ccache or sccache setup as building vLLM could
take sometimes.

### vLLM benchmark on PyTorch infra

* Run the benchmark on all vLLM main commits continuously

```
HF_TOKEN=<REDACTED> ./cron.sh
```

* Run the benchmark on the latest commit in a branch, i.e. `main`

```
HF_TOKEN=<REDACTED> ./run.sh main
```

* Run the benchmark on a specific commit on [vLLM](https://github.com/vllm-project/vllm)

```
HF_TOKEN=<REDACTED> ./run.sh <COMMIT_SHA>
```

* Run the benchmark, but don't upload the results to PyTorch OSS
  benchmark database

```
HF_TOKEN=<REDACTED> UPLOAD_BENCHMARK_RESULTS=0 ./run.sh main
```

* Run the benchmark on the commit even if it has already been run before

```
HF_TOKEN=<REDACTED> OVERWRITE_BENCHMARK_RESULTS=1 ./run.sh main
```

The results and other artifacts will be available at:

* Benchmark results in JSON: `https://ossci-benchmarks.s3.us-east-1.amazonaws.com/v3/vllm-project/vllm/<BRANCH>/<COMMIT>/benchmark_results.json`
* Benchmark results in markdown: `https://ossci-benchmarks.s3.us-east-1.amazonaws.com/v3/vllm-project/vllm/<BRANCH>/<COMMIT>/benchmark_results.md`
* Benchmark logs: `https://ossci-benchmarks.s3.us-east-1.amazonaws.com/v3/vllm-project/vllm/<BRANCH>/<COMMIT>/benchmarks.log`
