from mpi4py import MPI
import numpy as np

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

def p(x):
    print(np.round(x,2))

def gather_and_log():
    Qi_list = comm.gather(Qi,root=0)
    if rank == 0:
        Q = np.concatenate(Qi_list,axis=0)
        p(Q)

m = 11
n = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rng = None
if rank == 0:
    rng = np.random.default_rng()
rng = comm.bcast(rng,0)
A = rng.uniform(-2,2,(m,n))

# distribute matrices
block_size = int(np.ceil(m/size))
start_row = rank*block_size
end_row = min((rank+1)*block_size,m)
Ai = A[start_row:end_row,:].copy()
Qi = A[start_row:end_row,:].copy()
R = np.zeros((n,n))

beta_send = np.empty(1)
beta = np.empty(1)
r_send = np.empty(1)
r = np.empty(1)

for j in range(n):
    for i in range(j):
        r_send[0] = Qi[:,i] @ Qi[:,j]
        comm.Allreduce(r_send,r,op=MPI.SUM)
        R[i,j] = r[0]
        Qi[:,j] = Qi[:,j] - Qi[:,i] * R[i,j]
    beta_send[0] = np.sum(Qi[:,j]**2)
    comm.Allreduce(beta_send,beta,op=MPI.SUM)
    R[j,j] = np.sqrt(beta[0])
    Qi[:,j] = Qi[:,j]/R[j,j]


Qi_list = comm.gather(Qi,root=0)
if rank == 0:
    Q = np.concatenate(Qi_list,axis=0)
    Q_t,R_t = mgs(A)
    print("A")
    p(A)
    print("Q - Q_sequenciel")
    p(Q - Q_t)
    print("QR - A")
    p(Q@R - A)
    print("Q.TQ",(Q@Q.T).shape)
    p(Q.T@Q)
    print("R",R.shape)
    p(R)
    print("Q_tR_t - A",A.shape)
    p(Q_t@R_t - A)
    print("Q_t.TQ_t")
    p(Q_t.T@Q_t)
    print("R_t")
    p(R_t)

