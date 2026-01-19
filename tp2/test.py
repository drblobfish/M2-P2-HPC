from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 3
m = 3
A = None
Ai = np.empty(
if rank == 0:
    A = np.arange(n*m,dtype=np.double)
comm.Scatter(A,Ai,root=0)
print(rank,Ai)


