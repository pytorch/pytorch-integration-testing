#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.

set -eou pipefail

(
    set -x
    python -u run_test.py --continue-through-error -v
)

(
    set -x
    conda install -y tensorflow scipy
    python -u -c "import torch; import tensorflow"
    python -u -c "import tensorflow; import torch"
    python -u -c "import scipy; import torch"
    python -u -c "import torch; import scipy"
    python -u -c "import torch as th; x = th.autograd.Variable(th.rand(1, 3, 2, 2)); l = th.nn.Upsample(2); print(l(x))"
)
