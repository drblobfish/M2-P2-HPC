import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

def randomized_nystrom_rank_k(A,k,I):
    n = A.shape[0]
    assert A.shape[1] == n,"A must be square"
    rng = np.random.default_rng()
    # Omega is (n,I)
    Omega = rng.normal(0,1,(n,I))
    # step 1 : C is (n,I)
    C = A @ Omega
    #step 2 : B is (I,I)
    B = Omega.T @ C
    use_cholesky = True
    try :
        # L is (I,I) and lower triangular
        L = np.linalg.cholesky(B)
    except :
        # U,S,V are (I,I), since B is symmetric, U = V
        U,S,_ = np.linalg.svd(B,hermitian=True)
        use_cholesky = False
    # step 3 : Z is (n,I)
    if use_cholesky :
        # if we used cholesky, L is triangular
        Z = scipy.linalg.solve_triangular(L,C.T,lower=True).T
    else :
        # if we used SVD, we have a simple expression for L^{-T}
        Z = C @ U @ np.diag(1/np.sqrt(S))
    # step 4 : Q is (n,I); R is (I,I)
    Q,R = np.linalg.qr(Z)
    # step 5 : U,S are (I,I); Uk is (I,k); Sk is (k,k)
    U,S,_ = np.linalg.svd(R)
    Uk = U[:,:k]
    Sk = S[:k]
    # step 6 : Uhat is (n,k)
    Uhat = Q @ Uk
    # step 7 : Anyst is (n,n)
    Anyst = Uhat @ np.diag(Sk**2) @ Uhat.T
    return Anyst,use_cholesky

def nuclear_norm(A):
    return np.sum(np.linalg.svd(A,compute_uv = False))

def trace_relative_error(A,B):
    return nuclear_norm(A-B)/nuclear_norm(A)

def load_libsvm(file,nb_point,nb_features):
    mat = np.zeros((nb_point,nb_features))
    cls = np.empty(nb_point)
    with open(file,"r") as f:
        for i in range(nb_point):
            l = f.readline().strip().split(" ")
            cls[i] = int(l[0])
            for r in l[1:]:
                col,val = r.split(":")
                mat[i,int(col)] = float(val)
    return mat,cls

def prepare_dataset(A,c):
    n,k = A.shape
    B = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            B[i,j] = np.exp(-np.sum((A[i]-A[j])/c)**2)
    return B

# constants
n = 500
assert n<=60_000
c = 100
replication = 20

# k_scale = np.linspace(200,600,5,dtype=np.integer)
# I_scale = [600,1000,2000,2500,3000]
k_scale = np.linspace(20,60,5,dtype=np.integer)
I_scale = [60,100,200,250,300]

print("loading dataset")
mat,_ = load_libsvm("mnist.scale",n,781)
print("preparing dataset")
A = prepare_dataset(mat,c)
print(A)

result_df = pd.merge(pd.DataFrame({"I":I_scale}),pd.DataFrame({"k":k_scale}),how="cross").merge(pd.DataFrame({"replication":np.arange(replication,dtype = np.int32)}),how="cross")
result_df["trace_error"] = 0.0
result_df["use_cholesky"] = True;
for i in range(result_df.shape[0]):
    I = result_df["I"][i]
    k = result_df["k"][i]
    print(f"experiment I={I} k={k}")
    Anyst,use_cholesky = randomized_nystrom_rank_k(A,k,I)
    err = trace_relative_error(A,Anyst)
    result_df.loc[i,"trace_error"] = err
    result_df.loc[i,"use_cholesky"] = use_cholesky

aggregated_df = result_df.groupby(["I","k"])["trace_error"].mean().reset_index()
param_string = f"n = {n}, c = {c}, replication = {replication}"
sns.lineplot(aggregated_df,x="k",y="trace_error",hue="I").set(title = param_string)
plt.savefig("error_plot"+time.strftime("%m_%d_%H_%M")+".pdf")

plt.show()
