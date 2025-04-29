import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def laplacian_2d(nx, ny, dx, dy):
    # 1D Laplacian with periodic boundary condition (x-direction)
    main_diag_x = -2.0 * np.ones(nx)
    off_diag_x = np.ones(nx - 1)
    lap_1d_x = sp.diags([main_diag_x, off_diag_x, off_diag_x], [0, -1, 1], format='csr')
    lap_1d_x[0, -1] = 1.0
    lap_1d_x[-1, 0] = 1.0
    
    # 1D Laplacian with periodic boundary condition (y-direction)
    main_diag_y = -2.0 * np.ones(ny)
    off_diag_y = np.ones(ny - 1)
    lap_1d_y = sp.diags([main_diag_y, off_diag_y, off_diag_y], [0, -1, 1], format='csr')
    lap_1d_y[0, -1] = 1.0
    lap_1d_y[-1, 0] = 1.0
    
    # 2D Laplacian using Kronecker product
    I_x = sp.eye(nx, format='csr')
    I_y = sp.eye(ny, format='csr')
    lap_x = sp.kron(I_y, lap_1d_x/ dx**2) 
    lap_y = sp.kron(lap_1d_y, I_x/ dy**2)
    
    lap_2d = -0.5 * (lap_x/dx**2 + lap_y/dy**2)
    return lap_2d

def potential_2d(x, y):
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    V = -1.0 / r
    print(x[1]-x[0])
    return sp.diags(V.ravel(), format='csr')

def plot_wavefunction(psi, nx, ny, x, y):
    wavefunc = np.abs(psi.reshape((nx, ny)))**2
    plt.imshow(wavefunc, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(label="Probability Density")
    plt.title("Wavefunction Probability Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


n = 6
Lx = 4.0   
Ly = 4.0   
nx = 2**n    
ny = 2**n   

x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]


T = laplacian_2d(nx, ny, dx, dy)
V = potential_2d(x, y)


H = T + V


num_states = 5 
eigenvalues, eigenvectors = spla.eigsh(H, k=num_states, which='SA')



for i, val in enumerate(eigenvalues):
    print(f"Energy {i+1}: {val:.6f}")


plot_wavefunction(eigenvectors[:, 0], nx, ny, x, y)
plot_wavefunction(eigenvectors[:, 1], nx, ny, x, y)
