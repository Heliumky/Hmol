import plot_utility_jit as ptut
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np




input_mps_path = "3d_dmrg_results_N=8.pkl"
with open(input_mps_path, 'rb') as file:
    loaded_data = pickle.load(file)
#print(loaded_data)
N = loaded_data['N']
shift = loaded_data['shift']
rescale = loaded_data['rescale']
xmax = loaded_data['xmax']
cutoff = loaded_data['cutoff']
dx = loaded_data['dx']
energy_1s = loaded_data['energy_1s']
energy_2s = loaded_data['energy_2s']
phi0 = loaded_data['phi0']
phi1 = loaded_data['phi1']



# --------------- Plot ---------------------
bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))
bzs = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
zs = ptut.bin_to_dec_list (bzs,rescale=rescale, shift=shift)
X, Y, Z = np.meshgrid (xs, ys, zs)

#ZV = ptut.get_3D_mesh_eles_mps (V_MPS, bxs, bys, bzs)
Z0 = ptut.get_3D_mesh_eles_mps (phi0, bxs, bys, bzs)
Z1 = ptut.get_3D_mesh_eles_mps (phi1, bxs, bys, bzs)
#Z2 = ptut.get_3D_mesh_eles_mps (phi2, bxs, bys)
#Z3 = ptut.get_3D_mesh_eles_mps (phi3, bxs, bys)
INTZ_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=2) * dx
INTZ_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=2) * dx
INTX_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=0) * dx
INTX_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=0) * dx
INTY_Z0 = np.sum(np.abs(Z0 / dx**1.5)**2, axis=1) * dx
INTY_Z1 = np.sum(np.abs(Z1 / dx**1.5)**2, axis=1) * dx

# XY plane
# Ground state density
plt.figure(figsize=(12, 24))
plt.subplot(3, 2, 1)
plt.imshow(INTZ_Z0, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s XY Density (Energy = {energy_1s:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# Excited state density
plt.subplot(3, 2, 2)
plt.imshow(INTZ_Z1, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s XY Density (Energy = {energy_2s:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# XZ plane
# Ground state density
plt.subplot(3, 2, 3)
plt.imshow(INTY_Z0, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s XZ Density (Energy = {energy_1s:.8f})')
plt.xlabel('x')
plt.ylabel('z')

# Excited state density
plt.subplot(3, 2, 4)
plt.imshow(INTY_Z1, extent=(X.min(), X.max(), Z.min(), Z.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s XZ Density (Energy = {energy_2s:.8f})')
plt.xlabel('x')
plt.ylabel('z')

# YZ plane
# Ground state density
plt.subplot(3, 2, 5)
plt.imshow(INTX_Z0, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'1s YZ Density (Energy = {energy_1s:.8f})')
plt.xlabel('y')
plt.ylabel('z')

# Excited state density
plt.subplot(3, 2, 6)
plt.imshow(INTX_Z1, extent=(Y.min(), Y.max(), Z.min(), Z.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'2s YZ Density (Energy = {energy_2s:.8f})')
plt.xlabel('y')
plt.ylabel('z')

plt.tight_layout()
plt.savefig(f"3d_dmrg_density_functions_INT{N}.pdf", format='pdf')
plt.show()

'''
# Ground state density
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(Z0[:,:,2**N//2]/dx**1.5)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'Ground State Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# Excited state density
plt.subplot(1, 2, 2)
plt.imshow(np.abs(Z1[:,:,2**N//2]/dx**1.5)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'First Excited State Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig("3d_dmrg_density_functions.pdf", format='pdf')
plt.show()
'''

#fig = plt.figure(figsize=(12, 6))
#ax = fig.add_subplot(111, projection='3d')
#plt.imshow(np.abs(Z0/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
#sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z0.flatten(), cmap='viridis', s=1)
#plt.colorbar(sc, label='Intensity')
#ax.set_title(f'Ground State Density (Energy = {energy_ground:.8f})')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')

