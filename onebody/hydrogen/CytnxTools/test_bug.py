import sys
#import cytnx
import numpy as np
from ncon import ncon

def inner_MPS (mps1, mps2):
    assert len(mps1) == len(mps2)
    res = ncon([mps1[0], np.conj(mps2[0])], ((1,2,-1), (1,2,-2)))
    for i in range(1,len(mps1)):
        res = ncon([res,mps1[i],np.conj(mps2[i])], ((1,2), (1,3,-1), (2,3,-2)))
    res = res.reshape(-1)
    assert len(res) == 1
    return res[0]

def compress_MPS (mps, D=sys.maxsize, cutoff=0.):
    N = len(mps)
    #
    #        2 ---
    #            |
    #   R =      o
    #            |
    #        1 ---
    #
    Rs = [None for i in range(N+1)]
    Rs[-1] = np.array([1.]).reshape((1,1))
    for i in range(N-1, 0, -1):
        Rs[i] = ncon([Rs[i+1],mps[i],np.conjugate(mps[i])], ((1,2), (-1,3,1), (-2,3,2)))


    #
    #          2
    #          |
    #   rho =  o
    #          |
    #          1
    #
    rho = ncon([Rs[1],mps[0],np.conjugate(mps[0])], ((1,2), (-1,-2,1), (-3,-4,2)))
    rho = rho.reshape((rho.shape[1], rho.shape[3]))

    #         1
    #         |
    #    0 x--o-- 2
    #
    evals, U = np.linalg.eigh(rho)
    U = U.reshape((1,*U.shape))
    res = [U]

    #
    #        ---- 2
    #        |
    #   L =  |
    #        |
    #        ---- 1
    #
    L = np.array([1.]).reshape((1,1))
    for i in range(1,N):
        #
        #         2---(U)-- -2
        #         |    |
        #   L =  (L)   3
        #         |    |
        #         1---(A)--- -1
        #
        L = ncon([L,mps[i-1],np.conjugate(U)], ((1,2), (1,3,-1), (2,3,-2)))

        #
        #          -- -1
        #          |
        #   A =    |       -2
        #          |        |
        #         (L)--1--(mps)--- -3
        #
        A = ncon([L,mps[i]], ((1,-1), (1,-2,-3)))

        #
        #         -3 --(A)--2--
        #               |     |
        #              -4     |
        #   rho =            (R)
        #              -2     |
        #               |     |
        #         -1 --(A)--1--
        #
        rho = ncon([Rs[i+1],A,np.conjugate(A)], ((1,2), (-1,-2,1), (-3,-4,2)))
        d = rho.shape
        rho = rho.reshape((d[0]*d[1], d[2]*d[3]))

        #         1
        #         |
        #     0 --o-- 2
        #
        evals, U  = np.linalg.eigh(rho)
        # truncate by dimension
        DD = min(D, mps[i].shape[2])
        U = U[:,-DD:]
        evals = evals[-DD:]
        # truncate by cutoff
        iis = (evals > cutoff)
        U = U[:,iis]
        #
        U = U.reshape((d[2],d[3],U.shape[1]))
        res.append(U)
    return res

def random_MPS(N, phydim, seed, vdim=1, dtype=np.complex128):
    mps = []
    np.random.seed(seed)
    is_complex = np.issubdtype(dtype, np.complexfloating)
    for i in range(N):
        if i == 0:
            arr = np.random.rand(1, phydim, vdim).astype(dtype)
        elif i == N-1:
            arr = np.random.rand(vdim, phydim, 1).astype(dtype)
        else:
            arr = np.random.rand(vdim, phydim, vdim).astype(dtype)
        if is_complex:
            if i == 0:
                arr += 1j * np.random.rand(1, phydim, vdim).astype(dtype)
            elif i == N-1:
                arr += 1j * np.random.rand(vdim, phydim, 1).astype(dtype)
            else:
                arr += 1j * np.random.rand(vdim, phydim, vdim).astype(dtype)

        mps.append(arr)

    return mps

psi0 = random_MPS (12, 2, 15)
psi0 = compress_MPS (psi0, cutoff=1e-18)
print(inner_MPS(psi0,psi0))

