from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

whoseturn = np.zeros(1,dtype=int)
p = 0.9

while whoseturn[0] != -1:
    root = whoseturn[0]
    if whoseturn[0] == rank:
        if random.random()<p:
            print("ping")
            whoseturn[0] = random.randint(0,size-1)
        else:
            print("fin du jeu!")
            whoseturn[0] = -1
    comm.Bcast(whoseturn,root=root)

print("Ok on arrête")
