import numpy as np


def cgs(A):
    m,n = A.shape
    Q = np.empty((m,n))
    R = np.zeros((n,n))

    R[0,0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0]/R[0,0]

    for j in range(1,n):
        R[:j,j] = Q[:,:j].T @ A[:,j]
        Q[:,j] = A[:,j] - Q[:,:j] @ R[:j,j]
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] = Q[:,j ]/R[j,j]

    return Q,R

def mgs(A):
    m,n = A.shape
    Q = np.empty((m,n))
    Q[:,:] = A[:,:]
    R = np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j] = Q[:,i] @ Q[:,j]
            Q[:,j] = Q[:,j] - Q[:,i] * R[i,j]
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] = Q[:,j] / R[j,j]

    return Q,R

n = 5
m = 5
A = np.arange(n*m).reshape(m,n)
Q,R = mgs(A)
print(Q)
print(R)
print(np.round(Q@Q.T,3))
exit()

m = 50
n = 20
W = np.arange(1, m*n+1, 1, dtype = 'd').reshape((m, n))
W = W + np.eye(m, n) # Make it full rank

m = 4
n = 3
ep = 1e-12
W = np.array([[1, 1, 1], [ep, 0, 0], [0, ep, 0], [0, 0, ep]])
