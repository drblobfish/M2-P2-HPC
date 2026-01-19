from mpi4py import MPI
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

m = 5
n = 5
#A = np.empty((m,n))
A = np.arange(n*m,dtype=np.double).reshape(m,n)
Q = np.empty((m,n))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# distribute matrices
block_size = int(np.ceil(m/size))
start_row = rank*block_size
end_row = min((rank+1)*block_size,m)
Ai = A[start_row:end_row,:]
Qi = Q[start_row:end_row,:]
R = np.zeros((n,n))

beta_send = np.empty(1)
beta = np.empty(1)
beta_send[0] = np.sum(Ai[:,0]**2)
comm.Allreduce(beta_send,beta,op=MPI.SUM)
R[0,0] = np.sqrt(beta[0])
Qi[:,0] = Ai[:,0]/R[0,0]

for j in range(1,n):
    r_send = Qi[:,:j].T @ Ai[:,j]
    r = np.empty_like(r_send)
    comm.Allreduce(r_send,r,op=MPI.SUM)
    R[:j,j] = r
    Qi[:,j] = Ai[:,j] - Qi[:,:j] @ R[:j,j]
    beta_send[0] = np.sum(Qi[:,j]**2)
    comm.Allreduce(beta_send,beta,op=MPI.SUM)
    R[j,j] = np.sqrt(beta[0])
    Qi[:,j] = Qi[:,j]/R[j,j]

def p(x):
    print(np.round(x,2))

comm.Gather(Qi,Q,root=0)
if rank == 0:
    Q_t,R_t = cgs(A)
    print("A")
    p(A)
    print("Q")
    p(Q)
    print("R")
    p(R)
    print("Q_t")
    p(Q_t*Q_t.T)
    print("R_t")
    p(R_t)

