This directory mirrors the list of benchmarks from
[vLLM](https://github.com/vllm-project/vllm/tree/main/.buildkite/nightly-benchmarks/tests),
but it includes only models that we want to cover in PyTorch infra.

Another note is that speculative decoding is not yet supported in v1
with the exception of ngram, so its corresponding benchmarks is
currently removed from the list.
