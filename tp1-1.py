from mpi4py import MPI
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
c = np.zeros_like(a)
d = np.zeros_like(a)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    for i in range(4):
        c[i] = 2*a[i] + b[i]
    comm.Send(a, dest = 1, tag = 0)
elif rank == 1:
    comm.Recv(a, source = 0, tag = 0)
    for i in range(4):
        d[i] = 2*a[i] + b[i]
print("I am rank = ", rank, " and my d = ", d)
