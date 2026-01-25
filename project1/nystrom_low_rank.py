import numpy as np
import scipy
from mpi4py import MPI

def mpi_assert(cond,message = None):
    if cond :
        return
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("[ASSERT FAILED]",message)
    exit()

def mpi_qr(comm,Ai):
    """
    A is an n,n matrix that is stored as block of rows on the processors
    Ai is the block of rows of A owned by the processor
    returns Qi and R
    where Qi is the corresponding block of rows of Q
    where Q,R is the QR factorization of A, computed with modified Gram-Schmidt
    """
    (block_size,n) = Ai.shape
    Qi = Ai.copy()
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
    return Qi,R

# constants
k = 5
n = 100
I = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
P = comm.Get_size()

mpi_assert(np.sqrt(P).is_integer(), "number of precessor must be square")
sqrtP = int(np.sqrt(P))

mpi_assert(n % sqrtP == 0, "size of the matrix must be a multiple of number of processor")
block_size = int(n/sqrtP)

i = rank % sqrtP
j = rank // sqrtP

row_comm = comm.Split(i,j)
col1_comm = comm.Split(0 if j==0 else MPI.UNDEFINED,i)

# A_{kl} = kl
Aij = (np.arange(i*block_size,(i+1)*block_size,dtype=np.double).reshape(-1,1) *
       np.arange(j*block_size,(j+1)*block_size,dtype=np.double))

seeds = np.empty(P,dtype=np.uint32)
if rank == 0:
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=sqrtP, dtype=np.uint32)

comm.Bcast(seeds,0);

rng_i = np.random.default_rng(seeds[i])
rng_j = np.random.default_rng(seeds[j])
Omega_i = rng_i.normal(0,1,(block_size,I))
Omega_j = rng_j.normal(0,1,(block_size,I))
# step 1 : C is (n,I)
C_ij = Aij @ Omega_j
C_i = np.empty_like(C_ij)
row_comm.Reduce(C_ij,C_i,MPI.SUM,0)
# step 2 : B is (I,I); L is (I,I)
B_ij = Omega_i.T @ C_ij
B = np.empty_like(B_ij)
comm.Reduce(B_ij,B,MPI.SUM,0)
if j==0:
    Linv = np.empty_like(B_ij)
    if rank == 0:
        U,S,_ = np.linalg.svd(B,hermitian=True)
        Linv = U @ np.diag(1/np.sqrt(S))
    col1_comm.Bcast(Linv,0)
    # step 3 : Z is (n,I)
    Z_i = C_i @ Linv
    # step 4 : Q is (n,I); R is (I,I)
    Q_i,R = mpi_qr(col1_comm,Z_i)
    # step 5 : U,S are (I,I); Uk is (I,k); Sk is (k,k)
    U,S,_ = np.linalg.svd(R)
    Uk = U[:,:k]
    Sk = S[:k]
    # step 6 : Uhat is (n,k)
    Uhat_i = Q_i @ Uk
    # step 7 : Anyst is (n,n), we can compute its diagonal blocks easily
    Anyst_ii = Uhat_i @ np.diag(Sk**2) @ Uhat_i.T
    # at this point if we need the entirety of Anyst distributed on all P
    # processor like A, we need to do extra work. This might not be useful
    # depending on how we plan to use Anyst
