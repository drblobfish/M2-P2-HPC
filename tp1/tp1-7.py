from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

senddata = rank*np.arange(size, dtype = int)
global_result1 = np.empty(size, np.int64)
global_result3 = np.empty(size, np.int64)
comm.Reduce(senddata, global_result1, op = MPI.SUM, root = 0)
global_result2 = comm.reduce(rank, op = MPI.SUM, root = 0)
comm.Reduce(senddata, global_result3, op = MPI.MAX, root = 0)

print(" process ", rank, " sending ", senddata)
if rank == 0:
    print("Reduction operation1: ", global_result1)
    print("Reduction operation2: ", global_result2)
    print("Reduction operation3: ", global_result3)
