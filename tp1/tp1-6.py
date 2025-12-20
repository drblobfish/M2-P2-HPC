from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

senddata = rank*np.ones(size, dtype = int)
recvdata = np.empty(size, np.int64)
comm.Alltoall(senddata, recvdata)

print(" process ", rank, " sending ", senddata, " receiving ", recvdata )
