from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data1 = np.random.rand(size)
    data2 = np.random.rand(size)
else:
    data1 = np.empty(size)
    data2 = None

data3 = np.empty(1, np.float64)
comm.Bcast(data1, root = 0)
comm.Scatter(data2, data3, root = 0)
comm.Barrier()
print("rank: ", rank, " data1: ", data1, " data2: ", data2, " data3: ", data3, flush=True)
