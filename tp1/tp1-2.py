from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 5000

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    print("From process: ", rank, " data sent:", data, "\n")
    req = comm.isend(data, dest=1, tag=1)
elif rank == 1:
    req = comm.irecv(source=0, tag=1)
    data = req.wait()
    print("From process: ", rank, " data received:", data, "\n")
elif rank == 2:
    data = np.ones(N, np.int64)
    print("From process: ", rank, " data sent:", data, "\n")
    req = comm.Isend([data, N, MPI.LONG], dest=3, tag=2)
elif rank == 3:
    data = np.empty(N, np.int64)
    req = comm.Irecv([data, N, MPI.LONG], source=2, tag=2)
    req.wait()
    print("From process: ", rank, " data received:", data, "\n")


MPI.Request.Waitall
