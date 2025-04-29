import dmrg as dmrg
import numpy as np
import qtt_utility as ut
import os
import MPS_utility as mpsut
import hamilt
from test import load_mps
import npmps
import pickle
import plot_utility as ptut
import matplotlib.pyplot  as plt

if __name__ == '__main__':
    N = 9
    #cutoff = 0.1
    shift = - 5
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)



    # Generate the MPS for the potential


    # Kinetic energy
    Hk1, Lk1, Rk1 = hamilt.H_kinetic(N, dx)
    Hk, Lk, Rk = hamilt.get_H_2D (Hk1, Lk1, Rk1)
    assert len(Hk) == 2*N

    # Potential energy
    factor = 1
    os.system('python3 tci.py '+str(2*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --2D_one_over_r')
    V_MPS = load_mps(f'fit{2*N}.mps.npy')
    #V_MPS = tci.tci_one_over_r_2D (2*N, rescale, cutoff, factor, shift)
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    '''bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs)
    ys = ptut.bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)
    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, ZV, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()'''

    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)
    assert len(H) == 2*N

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Create MPS


    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*100 + [32]*100
    cutoff = 1e-12

    # Run dmrg
    maxdims = [2]*20 + [4]*50 + [8]*150 + [16]*200 + [32]*50
    psi0 = npmps.random_MPS (2*N, 2, 2)
    psi0 = npmps.compress_MPS (psi0, cutoff=1e-18)
    print(npmps.inner_MPS(psi0,psi0))
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (psi0, H, L, R, maxdims, cutoff)
 
    np.savetxt(f'2d_terr0N={N}.dat',(terrs0,ens0))

    #maxdims = maxdims + [64]*100
    psi1 = npmps.random_MPS (2*N, 2, 2)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[20])
    np.savetxt(f'2d_terr1={N}.dat',(terrs1,ens1))

    #maxdims = maxdims + [64]*256
    #psi2 = npmps.random_MPS (2*N, 2, 2)
    #psi2 = mpsut.npmps_to_uniten (psi2)
    #psi2,ens2,terrs2 = dmrg.dmrg (psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    #np.savetxt('terr2.dat',(terrs2,ens2))

    #psi3 = npmps.random_MPS (2*N, 2, 2)
    #psi3 = mpsut.npmps_to_uniten (psi3)
    #psi3,ens3,terrs3 = dmrg.dmrg (psi3, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2], weights=[20,20,20])
    #np.savetxt('terr3.dat',(terrs3,ens3))

    #print(ens0[-1], ens1[-1], ens2[-1], ens3[-1])
    print(ens0[-1], ens1[-1])
    energy_ground = ens0[-1]
    energy_excited = ens1[-1]

    phi0 = mpsut.to_npMPS (psi0)
    print(npmps.inner_MPS(phi0 ,phi0 ))
    phi1 = mpsut.to_npMPS (psi1)
    #phi2 = mpsut.to_npMPS (psi2)
    #phi3 = mpsut.to_npMPS (psi3)

    data = {
        'N': N,
        'shift': shift,
        'rescale': rescale, 
        'xmax': xmax,  
        'cutoff': rescale, 
        'dx': dx, 
        'energy_1s': energy_1s,
        'energy_2s': energy_2s,
        'phi0': phi0,
        'phi1': phi1,
        'psi0': psi0,
        'psi1': psi1,
        'V_MPS': V_MPS,
    }
    with open(f'2d_dmrg_results_N={N}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    bys = list(ptut.BinaryNumbers(N))

    xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)
    ys = ptut.bin_to_dec_list (bys,rescale=rescale, shift=shift)
    X, Y = np.meshgrid (xs, ys)

    ZV = ptut.get_2D_mesh_eles_mps (V_MPS, bxs, bys)
    Z0 = ptut.get_2D_mesh_eles_mps (phi0, bxs, bys)
    Z1 = ptut.get_2D_mesh_eles_mps (phi1, bxs, bys)
    #Z2 = ptut.get_2D_mesh_eles_mps (phi2, bxs, bys)
    #Z3 = ptut.get_2D_mesh_eles_mps (phi3, bxs, bys)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z0, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface (X, Y, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Ground state density
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(Z0/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title(f'Ground State Density (Energy = {energy_ground:.8f})')
    plt.xlabel('x')
    plt.ylabel('y')

# Excited state density
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(Z1/dx)**2, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='plasma')
    plt.colorbar(label='Intensity')
    plt.title(f'First Excited State Density (Energy = {energy_excited:.8f})')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f"2d_dmrg_density_functions{N}.pdf", format='pdf')
    plt.show()

