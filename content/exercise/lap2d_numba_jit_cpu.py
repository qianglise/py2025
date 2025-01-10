import numpy as np
import numba 

@numba.jit
def lap2d_numba_jit_cpu(u, unew):
    M, N = u.shape   
    for i in range(1, M-1):
        for j in range(1, N-1):             
            unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )     


# Benchmark

M = 4096
N = 4096

u = np.zeros((M, N), dtype=np.float64)
unew = np.zeros((M, N), dtype=np.float64)

%timeit lap2d_numba_jit_cpu(u, unew)
