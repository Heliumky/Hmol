import numpy as np
import npmps
from numba import njit, prange
from tci import load_mps
import os


@njit
def dec_to_bin(dec, N):
    bstr = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        bstr[N - 1 - i] = (dec >> i) & 1
    return bstr

@njit
def bin_array_to_dec(bstr, rescale=1.0, shift=0.0):
    dec = 0
    for i in range(len(bstr)):
        dec += bstr[i] * (2 ** i)
    return dec * rescale + shift

@njit(parallel=True)
def bin_to_dec_list(bstr, rescale=1.0, shift=0.0):
    dec_list = np.zeros(len(bstr), dtype=np.float64)
    for i in prange(len(bstr)):
        dec_list[i] = bin_array_to_dec(bstr[i], rescale, shift)
    return dec_list

# An iterator for binary numbers
class BinaryNumbers:
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin(dec, self.N_num)[::-1]
        else:
            raise StopIteration


@njit
def get_ele_mps(mps, bstr):
    assert len(mps) == len(bstr)
    res = np.array([[1.]])
    for i in range(len(mps)):  # Avoid parallelizing this loop for safety
        A = mps[i].astype(np.float64)  # Ensure dtype is np.float64
        bi = bstr[i]
        M = np.ascontiguousarray(A[:, bi, :])
        res = np.dot(res, M)  # Consider using np.matmul or direct dot product
    return res[0][0]


@njit
def get_ele_mpo(mpo, L, R, bstr):
    mpo = npmps.absort_LR(mpo, L, R)
    res = np.array([[1.]])
    for i in range(len(mpo)):  # Parallelize this loop
        A = mpo[i]
        bi = int(bstr[i])
        M = A[:, bi, bi, :]
        res = np.dot(res, M)
    return float(res)

@njit(parallel=True)
def get_2D_mesh_eles_mps(mps, bxs, bys):
    nx, ny = len(bxs), len(bys)
    fs = np.zeros((nx, ny), dtype=np.float64)
    for i in prange(nx):  # Parallelize this loop
        for j in prange(ny):
            bstr = np.hstack((bxs[i], bys[j]))  # Directly concatenate arrays
            fs[i, j] = get_ele_mps(mps, bstr)
    return fs

@njit(parallel=True)
def get_3D_mesh_eles_mps(mps, bxs, bys, bzs):
    nx, ny, nz = len(bxs), len(bys), len(bzs)
    fs = np.zeros((nx, ny, nz), dtype=np.float64)
    for i in prange(nx):  # Parallelize this loop
        for j in prange(ny):
            for k in prange(nz):
                bstr = np.hstack((bxs[i], bys[j], bzs[k]))  # Directly concatenate arrays
                #print(bstr)
                fs[i, j, k] = get_ele_mps(mps, bstr)
    return fs

# Function for 2D mesh MPO
@njit(parallel=True)
def get_2D_mesh_eles_mpo(mpo, L, R, bxs, bys):
    nx, ny = len(bxs), len(bys)
    fs = np.zeros((nx, ny), dtype=np.float64)
    for i in prange(nx):  # Parallelize this loop
        for j in prange(ny):
            bstr = bxs[i] + bys[j]  # Combine binary numbers directly
            fs[i, j] = get_ele_mpo(mpo, L, R, bstr)
    return fs

if __name__ == '__main__':
    N = 3
    shift = -6
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =', xmax)
    print('xshift =', shift)
    print('dx =', dx)

    bxs = list(BinaryNumbers(N))
    bys = list(BinaryNumbers(N))
    bzs = list(BinaryNumbers(N))

    xs = bin_to_dec_list(bxs, rescale=rescale, shift=shift)
    ys = bin_to_dec_list(bys, rescale=rescale, shift=shift)
    zs = bin_to_dec_list(bzs, rescale=rescale, shift=shift)
    X, Y, Z = np.meshgrid(xs, ys, zs)

    factor = 1
    os.system('python3 tci.py ' + str(3*N) + ' ' + str(rescale) + ' ' + str(shift) + ' ' + str(cutoff) + ' ' + str(factor) + ' --3D_one_over_r')
    V_MPS = load_mps(f'fit{3*N}.mps.npy')

    #print(get_ele_mps(V_MPS, bstr=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])))
    ZV = get_3D_mesh_eles_mps(V_MPS, bxs, bys, bzs)
    INTZ_Z0 = np.sum(np.abs(ZV / dx**1.5)**2, axis=2) * dx

    import matplotlib.pyplot as plt
# XY plane
    # Ground state density
    plt.figure(figsize=(12, 24))
    plt.subplot(3, 2, 1)
    plt.imshow(INTZ_Z0, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    #plt.title(f'1s XY Density (Energy = {energy_excited:.8f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

