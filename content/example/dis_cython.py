%%cython

import numpy as np

def dis_cython(double[:,:] X):
    cdef ssize_t i,j,k,M,N
    cdef double d,tmp
    cdef double[:,:] D
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
