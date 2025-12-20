from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

p = 0.9

playing = True

if rank == 0:
    #send ball initially
    print("ping")
    req = comm.isend(True, dest=1)

while playing:
    ball = comm.recv()
    if ball :
        if random.random() < p:
            print("ping" if rank==0 else "pong")
            comm.send(True,dest=1-rank)
        else :
            playing = False
            # say it's the end of the game to the other
            print("Fin du jeu !!!")
            comm.send(False,dest=1-rank)
    else :
        print("ok j'arrête, c'etait super :)")
        playing = False



