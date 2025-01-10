def lap2d(u, unew):
    M, N = u.shape     
    for i in range(1, M-1):
        for j in range(1, N-1):             
            unew[i, j] = 0.25 * ( u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] )
