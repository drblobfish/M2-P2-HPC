from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
m = size
n = 10

Outbuf = None
A = None
v = None
if rank == 0:
    v = np.random.random(n)
    A = np.random.random([m,n])
    Outbuf = np.empty(1)
else :
    v = np.empty(n)

comm.Bcast(v,root=0)
Ai = np.empty(n)
comm.Scatter(A,Ai,root=0)
sendbuf = np.empty(1)
sendbuf[0] = (Ai @ v)**2
comm.Reduce(sendbuf,Outbuf,op = MPI.SUM, root=0)

if rank == 0:
    print(np.sqrt(Outbuf[0]))
    print(np.linalg.norm(A@v))

