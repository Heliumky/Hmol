import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import matplotlib.pyplot as plt
import pickle
import plot_utility_jit as ptut
import npmps
import numpy as np


#print(ptut.bin_to_dec_list (ptut.dec_to_bin(3, 5), rescale=1.0, shift=0.0))
#print(type(ptutj.dec_to_bin(3, 5)))  # 调试代码，查看 bstr 的类型
#print(ptutj.dec_to_bin(3, 5).shape[0])
#print(ptutj.bin_to_dec_list (ptutj.dec_to_bin(3, 5), rescale=1.0, shift=0.0))

input_mps_path = "2d_dmrg_results_N=8.pkl"

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


#phi0 = npmps.kill_site(phi0, sysdim=2, dtype=phi0[0].dtype)
#phi1 = npmps.kill_site(phi1, sysdim=2, dtype=phi1[0].dtype)
#print(npmps.inner_MPS(phi0,phi0))


phi0 = npmps.grow_site_0th(phi0, sysdim=2, dtype=phi0[0].dtype)
phi1 = npmps.grow_site_0th(phi1, sysdim=2, dtype=phi1[0].dtype)
print(npmps.inner_MPS(phi0,phi0))

phi0 = npmps.kill_site(phi0, sysdim=2, dtype=phi0[0].dtype)
phi1 = npmps.kill_site(phi1, sysdim=2, dtype=phi1[0].dtype)

print(npmps.inner_MPS(phi0,phi0))

#print([phi0[i].shape for i in range(len(phi0))])

#print(phi0[5])
#phi0 = mpsut.grow_site_2D(phi0)
#phi1 = mpsut.grow_site_2D(phi1)

# --------------- Plot ---------------------

#dx = dx/2

#bxs = list(ptut.BinaryNumbers(N+1))
#bys = list(ptut.BinaryNumbers(N+1))

bxs = list(ptut.BinaryNumbers(N))
bys = list(ptut.BinaryNumbers(N))

xs = ptut.bin_to_dec_list (bxs,rescale=dx, shift=shift)
ys = ptut.bin_to_dec_list (bys,rescale=dx, shift=shift)
X, Y = np.meshgrid (xs, ys)



#ZV = ptut.get_3D_mesh_eles_mps (V_MPS, bxs, bys, bzs)
Z0 = ptut.get_2D_mesh_eles_mps (phi0, bxs, bys)
Z1 = ptut.get_2D_mesh_eles_mps (phi1, bxs, bys)
#V_MPS = tci.load_mps(f'fit{2*N}.mps.npy')
#Z2 = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)


plt.figure(figsize=(8, 12))

wavefunctions = [Z0, Z1]
eigenvalues = [energy_1s, energy_2s]
titles = ['1s', '2?']
cmaps = ['viridis', 'plasma']

for i, (Z, title, cmap) in enumerate(zip(wavefunctions, titles, cmaps)):
    # real
    plt.subplot(3, 2, i + 1)
    plt.imshow(Z.real/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Re(\psi)$')
    plt.title(f'{title} Real Part\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # imag
    plt.subplot(3, 2, i + 3)
    plt.imshow(Z.imag/dx, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$Im(\psi)$')
    plt.title(f'{title} Imag Part\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # density
    plt.subplot(3, 2, i + 5)
    plt.imshow(np.abs(Z/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    plt.colorbar(label=f'$|\psi|^2$')
    plt.title(f'{title} Density\nEnergy = {eigenvalues[i]:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
plt.tight_layout()
plt.savefig(f"2d_extrp_{N}.pdf", format='pdf')
plt.show()

