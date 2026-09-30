"""Pin BLAS/OpenMP threads to 1 for the test run (shared server: parallelism only via workers).

pytest loads this file before the test modules, i.e. before numpy is imported; an explicit
shell export still wins (setdefault).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
