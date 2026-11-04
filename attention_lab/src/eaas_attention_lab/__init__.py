"""Look at the real weights, embeddings and attention matrices of GPT-2 small."""

import os

# NumPy hands matrix products to a BLAS library, which by default starts one
# thread per logical CPU. For matrices as small as ours, that pays off only on
# an idle computer. On a laptop with 20 logical CPUs, a forward pass of six
# tokens took about 0.05 s with four threads, and about as long with the
# default 20 threads while the laptop was idle. While other programs kept the
# CPU busy, the default often took seconds per pass, four threads well under
# one. Six or eight threads were about as fast as four. This line runs before
# NumPy is imported, because every script, and tests/conftest.py, imports this
# package first. A value you set yourself in the environment wins. Where NumPy
# uses another BLAS library, as on recent Macs, it changes nothing.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
