from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N = comm.Get_size()

p = 0.9

playing = True

if rank == 0:
    #send ball initially
    print("service")
    req = comm.isend(True, dest=random.randint(0,N-1))

while playing:
    ball = comm.recv()
    if ball :
        if random.random() < p:
            print(f"ping {rank}")
            comm.send(True,dest=random.randint(0,N-1))
        else :
            playing = False
            # say it's the end of the game to the other
            print("Fin du jeu !!!")
            for i in range(N):
                if i != rank:
                    comm.send(False,dest=i)
    else :
        playing = False
        print(f"ok j'arrête, c'etait super :) {rank}")



