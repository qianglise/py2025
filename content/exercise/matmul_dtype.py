import numpy as np
import numba
import numba.cuda

@numba.cuda.jit
def matmul_kernel(A, B, C):
    i, j = numba.cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# Benchmark

# first generate double precision input data

N = 8192
A = np.random.rand(N,N)
B = np.random.rand(N,N)
C = np.random.rand(N,N)

# copy them to GPU

d_A = numba.cuda.to_device(A)
d_B = numba.cuda.to_device(B)
d_C = numba.cuda.to_device(C)

# setup grid and block

threadsperblock = (16, 16)
blockspergrid = (10,10)

# benchmark double precision input data

%timeit matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C); numba.cuda.synchronize()

# then generate single precision input data

d_A32 = numba.cuda.to_device(A.astype(np.float32))
d_B32 = numba.cuda.to_device(B.astype(np.float32))
d_C32 = numba.cuda.to_device(C.astype(np.float32))

# benchmark single precision input data

%timeit matmul_kernel[blockspergrid, threadsperblock](d_A32, d_B32, d_C32); numba.cuda.synchronize()
