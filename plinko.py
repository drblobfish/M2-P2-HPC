from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def plinko_reduce(x,y):
    return x if random.random()<0.5 else y

s = comm.reduce(rank, op = plinko_reduce, root=0)

if rank == 0:
    print(s)

