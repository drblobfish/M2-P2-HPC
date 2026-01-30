#! /usr/bin/python3
from math import pi

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def mesh(nx,ny,Lx,Ly):
   i = np.arange(0,nx)[na,:] * np.ones((ny,1), np.int64)
   j = np.arange(0,ny)[:,na] * np.ones((1,nx), np.int64)
   p = np.zeros((2,ny-1,nx-1,3), np.int64)
   q = i+nx*j
   p[:,:,:,0] = q[:-1,:-1]
   p[0,:,:,1] = q[1: ,1: ]
   p[0,:,:,2] = q[1: ,:-1]
   p[1,:,:,1] = q[:-1,1: ]
   p[1,:,:,2] = q[1: ,1: ]
   v = np.concatenate(((Lx/(nx-1)*i)[:,:,na], (Ly/(ny-1)*j)[:,:,na]), axis=2)
   vtx = np.reshape(v, (nx*ny,2))
   elt = np.reshape(p, (2*(nx-1)*(ny-1),3))
   return vtx, elt 

def local_mesh(nx,ny,Lx,Ly,j,J):
    local_ny = ((ny-1)//J)+1
    local_Ly = Ly/J
    vtx, elt =  mesh(nx,local_ny,Lx,local_Ly)
    vtx[:,1] += j*local_Ly
    return vtx, elt

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(ny-1),nx*ny-1,1)[:,na],
                        np.arange(nx*(ny-1)+1,nx*ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(ny-1),nx)[:,na],
                        np.arange(nx,nx*ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*ny,nx)[:,na]))
    return np.vstack((bottom, top, left, right))

def local_boundary(nx,ny,j,J):
    local_ny = ((ny-1)//J)+1
    local_Ly = Ly/J
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(local_ny-1),nx*local_ny-1,1)[:,na],
                        np.arange(nx*(local_ny-1)+1,nx*local_ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(local_ny-1),nx)[:,na],
                        np.arange(nx,nx*local_ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(local_ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*local_ny,nx)[:,na]))
    if j == 0:
        beltj_artf = top
    elif j == J-1:
        beltj_artf = bottom
    else :
        beltj_artf = np.vstack((bottom, top))
    beltj_phys = np.vstack((bottom, top, left, right))
    return beltj_phys,beltj_artf

def Rj_matrix(nx,ny,j,J):
    local_ny = ((ny-1)//J)+1
    block_size = nx*local_ny
    block_shift = nx * (local_ny-1)
    Rj = np.zeros((block_size,nx*ny))
    Rj[:,j*block_shift:j*block_shift + block_size] = np.eye(block_size)
    return csr_matrix(Rj)

def Bj_matrix(nx,ny,j,J):
    local_ny = ((ny-1)//J)+1
    block_size = nx*local_ny
    if j == 0:
        Bj = np.zeros((nx,block_size))
        Bj[:,-nx:] = np.eye(nx)
    elif j == J-1:
        Bj = np.zeros((nx,block_size))
        Bj[:,:nx] = np.eye(nx)
    else :
        Bj = np.zeros((2*nx,block_size))
        Bj[:nx,:nx] = np.eye(nx)
        Bj[nx:,-nx:] = np.eye(nx)
    return csr_matrix(Bj)

def Cj_matrix(nx,ny,j,J):
    nb_interface = J-1
    S_size = 2*nx*nb_interface
    if j == 0:
        Cj = np.zeros((nx,S_size))
        Cj[:,:nx] = np.eye(nx)
    elif j == J-1:
        Cj = np.zeros((nx,S_size))
        Cj[:,(2*j-1)*nx:] = np.eye(nx)
    else :
        Cj = np.zeros((2*nx,S_size))
        Cj[:,(2*j-1)*nx:(2*j+1)*nx] = np.eye(2*nx)
    return csr_matrix(Cj)

def get_area(vtx, elt):
    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0])
    return areas

def mass(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    M = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = areas * (1 + (j == k)) / (d*(d+1))
           M += csr_matrix((val, (row, col)), shape=(nv, nv))
    return M

def stiffness(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    ne, d = np.shape(elt)
    E = np.empty((ne, d, d-1), dtype=np.float64)
    E[:,0,:] = 0.5 * (vtx[elt[:,1],0:2] - vtx[elt[:,2],0:2])
    E[:,1,:] = 0.5 * (vtx[elt[:,2],0:2] - vtx[elt[:,0],0:2])
    E[:,2,:] = 0.5 * (vtx[elt[:,0],0:2] - vtx[elt[:,1],0:2])
    K = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = np.sum(E[:,j,:] * E[:,k,:], axis=1) / areas
           K += csr_matrix((val, (row, col)), shape=(nv, nv))
    return K

def Aj_matrix(vtx,elt,beltj_phys,k):
    M = mass(vtx, elt)
    Mb = mass(vtx, beltj_phys)
    K = stiffness(vtx, elt)
    Aj = K - k**2 * M - 1j * k * Mb
    return Aj

def Tj_matrix(vtx,beltj_artf,Bj,k):
    Tj = k * Bj @ mass(vtx,beltj_artf) @ Bj.T
    return Tj

def Sj_factorization(Aj,Tj,Bj):
    local_problem_matrix = Aj - 1j * Bj.T @ Tj @ Bj
    fact = spla.splu(local_problem_matrix)
    return fact

def point_source(sp, k):    
    def ps(x):
        v = np.zeros(np.size(x,0), float)
        for s in sp:
            v += s[2]*np.exp(-10*(k/(2.0*pi))**2 * la.norm(x - s[na,0:2], axis=1)**2)
        return v
    return ps 

def bj_vector(vtx,elt,sp,k):
    M = mass(vtx, elt)
    b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
    return b

def S_operator(fact,B,T,C):
    def S(x):
        res = x.astype(np.complex128).copy() # identity
        for i,(factj,Bj,Tj,Cj) in enumerate(zip(fact,B,T,C)):
            res += 2*1j* Cj.T@Bj@(factj.solve(Bj.T @ Tj @ Cj @ x))
        return res
    return S

def Pi_operator(nx,J):
    nb_interface = J-1
    S_size = 2*nx*nb_interface
    # x is a vector of size 
    def Pi(x) :
        return x.reshape(nx,2,nb_interface)[:,(1,0),:].reshape(-1)
    return Pi
    
def g_vector(fact,B,C,b,Pi):
    S_size = C[0].shape[1]
    intermediate_res = np.zeros(S_size,dtype=np.complex128)
    for i,(factj,Bj,Cj,bj) in enumerate(zip(fact,B,C,b)):
        intermediate_res -= 2j * Cj.T @ Bj @ (factj.solve(bj))
    return Pi(intermediate_res)

def solve_fixed_point(Pi,S,g,relax):
    p = np.zeros_like(g)
    for i in range(nb_iter):
        p = (1-relax)*p - relax*Pi(S(p)) + relax * g
    return p


def plot_mesh(vtx, elt, val=None, **kwargs):
    trig = mtri.Triangulation(vtx[:,0], vtx[:,1], elt)
    if val is None:
        plt.triplot(trig, **kwargs)
    else:
        plt.tripcolor(trig, val,
                      shading='gouraud',
                      cmap=cm.jet, **kwargs)
    plt.axis('equal')

Lx = 1           # Length in x direction
Ly = 1           # Length in y direction
nx = 2           # Number of points in x direction
ny = 7           # Number of points in y direction
k  = 1           # Wavenumber of the problem
ns = 1           # Number of point sources + random position and weight below
J  = 3           # Number of domains

j = 0

sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
fact = []
B = []
T = []
C = []
b = []
for j in range(J):
    vtx,elt = local_mesh(nx,ny,Lx,Ly,j,J)
    beltj_phys,beltj_artf = local_boundary(nx,ny,j,J)
    Rj = Rj_matrix(nx,ny,j,J)
    Bj = Bj_matrix(nx,ny,j,J)
    Cj = Cj_matrix(nx,ny,j,J)
    Aj = Aj_matrix(vtx,elt,beltj_phys,k)
    Tj = Tj_matrix(vtx,beltj_artf,Bj,k)
    factj = Sj_factorization(Aj,Tj,Bj)
    bj = bj_vector(vtx,elt,sp,k)
    fact.append(factj)
    B.append(Bj)
    T.append(Tj)
    C.append(Cj)
    b.append(bj)

S = S_operator(fact,B,T,C)
Pi = Pi_operator(nx,J)
g = g_vector(fact,B,C,b,Pi)

p = np.zeros_like(g)
for i in range(nb_iter):
    p = (1-relax)*p - relax*Pi(S(p)) + relax * g
    u = 
return p

exit()
## Example resolution of model problem
Lx = 1           # Length in x direction
Ly = 2           # Length in y direction
nx = 1 + Lx * 32 # Number of points in x direction
ny = 1 + Ly * 32 # Number of points in y direction
k = 16           # Wavenumber of the problem
ns = 8           # Number of point sources + random position and weight below
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
vtx, elt = mesh(nx, ny, Lx, Ly)
belt = boundary(nx, ny)
M = mass(vtx, elt)
Mb = mass(vtx, belt)
K = stiffness(vtx, elt)
A = K - k**2 * M - 1j*k*Mb      # matrix of linear system 
b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
x = spla.spsolve(A, b)          # solution of linear system via direct solver

# GMRES
residuals = [] # storage of GMRES residual history
def callback(x):
    residuals.append(x)
y, _ = spla.gmres(A, b, rtol=1e-12, callback=callback, callback_type='pr_norm', maxiter=200)
print("Total number of GMRES iterations = ", len(residuals))
print("Direct vs GMRES error            = ", la.norm(y - x))

# Plots
plot_mesh(vtx, elt) # slow for fine meshes
plt.show()
plot_mesh(vtx, elt, np.real(x))
plt.colorbar()
plt.show()
plot_mesh(vtx, elt, np.abs(x))
plt.colorbar()
plt.show()
plt.semilogy(residuals)
plt.show()
