from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10000
p = np.random.random([2,N])
localpi = 4*((p**2).sum(0)<1).sum()/N
s = comm.reduce(localpi,MPI.SUM,root=0)

if rank == 0:
    pi = s/size
    print(f"pi={pi}")

