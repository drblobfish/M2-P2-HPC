import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

def randomized_nystrom_rank_k(A,k,I):
    n = A.shape[0]
    assert A.shape[1] == n,"A must be square"
    rng_i = np.random.default_rng()
    # Omega is (n,I)
    Omega = rng_i.normal(0,1,(n,I))
    # step 1 : C is (n,I)
    C = A @ Omega
    #step 2 : B is (I,I)
    B = Omega.T @ C
    use_cholesky = True
    try :
        # L is (I,I)
        L = np.linalg.cholesky(B)
    except :
        # U,S,V are (I,I)
        U,S,Vh = np.linalg.svd(B)
        use_cholesky = False
    if use_cholesky :
        # step 3 : Z is (n,I)
        Z = scipy.linalg.solve_triangular(L,C.T).T
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
    else :
        # step 5 : Uk is (I,k); Sk is (k,k) ; Vhk is (k,I)
        Uk = U[:,:k]
        Sk = S[:k]
        Vhk = U[:k,:]
        Anyst = C @ Vhk.T.conj() @ np.diag(1/Sk) @ Uk.T.conj() @ C.T
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
n = 4000
assert n<=60_000
c = 100

k_scale = np.linspace(200,600,5,dtype=np.integer)
I_scale = [600,1000,2000,2500,3000]
# k_scale = np.linspace(20,60,5,dtype=np.integer)
# I_scale = [60,100,200,250,300]

print("loading dataset")
mat,_ = load_libsvm("mnist.scale",n,781)
print("preparing dataset")
A = prepare_dataset(mat,c)
print(A)

result_df = pd.merge(pd.DataFrame({"I":I_scale}),pd.DataFrame({"k":k_scale}),how="cross")
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
sns.lineplot(result_df,x="k",y="trace_error",hue="I")
plt.savefig("error_plot"+time.strftime("%m_%d_%H_%M")+".pdf")

plt.show()
