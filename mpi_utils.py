import numpy as np
from mpi4py import MPI


a = np.arange(5*7).reshape(5,7)
a.shape
(row,col) = a.shape
rank = comm.Get_rank()
size = comm.Get_size()

block_size = int(np.ceil(row/size))
last_block = row - (size-1) * block_size
counts = np.repeat(block_size,size)
counts[size-1] = last_block
displ = np.cumsum(counts)

