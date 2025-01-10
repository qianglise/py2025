import numpy as np
import numba
N = 50
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

# matmul_numba_gpu.max_blocksize = 32 # may need to set it

%timeit matmul_numba_gpu(A, B, C)
# 10.9 ms ± 232 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
