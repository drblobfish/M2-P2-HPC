import numpy as np


def cgs(A):
    m,n = A.shape
    Q = np.empty_like(A)
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
    Q = A.copy()
    R = np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j] = Q[:,i] @ Q[:,j]
            Q[:,j] = Q[:,j] - Q[:,i] * R[i,j]
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] = Q[:,j] / R[j,j]

    return Q,R
