import dmrg as dmrg
import numpy as np
import qtt_utility as ut
import os
import MPS_utility as mpsut
import hamilt
from test import load_mps
import npmps
import pickle
#import tci
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    N = 4
    #cutoff = 0.1
    shift = - 6
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
    Hk, Lk, Rk = hamilt.get_H_3D (Hk1, Lk1, Rk1)
    assert len(Hk) == 3*N

    # Potential energy
    factor = 1
    os.system('python3 tci.py '+str(3*N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --3D_one_over_r')
    V_MPS = load_mps('fit.mps.npy')
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
    assert len(H) == 3*N

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Create MPS


    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    #maxdims = [2]*10 + [4]*10 + [8]*40 + [16]*40 + [32]*40
    cutoff = 1e-12

    # Run dmrg
    maxdims = [2]*10 + [4]*40 + [8]*80 + [16]*80 + [32]*80
    #maxdims = [2]*10
    psi0 = npmps.random_MPS (3*N, 2, 2)
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (psi0, H, L, R, maxdims, cutoff)
    np.savetxt(f'3d_terr0N={N}.dat',(maxdims,terrs0,ens0))

    maxdims = maxdims + [64]*80
    #maxdims = [2]*10 + [4]*40 + [8]*120 + [16]*120 + [32]*180
    #maxdims = [2]*10
    psi1 = npmps.random_MPS (3*N, 2, 2)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[20])
    np.savetxt(f'3d_terr1.dat',(maxdims,terrs1,ens1))


    #maxdims = maxdims + [64]*256
    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    #psi2 = npmps.random_MPS (3*N, 2, 2)
    #psi2 = mpsut.npmps_to_uniten (psi2)
    #psi2,ens2,terrs2 = dmrg.dmrg (psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    #np.savetxt('terr2.dat',(maxdims,terrs2,ens2))

    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    #psi3 = npmps.random_MPS (3*N, 2, 2)
    #psi3 = mpsut.npmps_to_uniten (psi3)
    #psi3,ens3,terrs3 = dmrg.dmrg (psi3, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2], weights=[20,20,20])
    #np.savetxt('terr3.dat',(maxdims,terrs3,ens3))

    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    #psi4 = npmps.random_MPS (3*N, 2, 2)
    #psi4 = mpsut.npmps_to_uniten (psi4)
    #psi4,ens4,terrs4 = dmrg.dmrg (psi4, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3], weights=[20,20,20,20])
    #np.savetxt('terr4.dat',(maxdims,terrs4,ens4))

    #maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*80 + [32]*80
    #psi5 = npmps.random_MPS (3*N, 2, 2)
    #psi5 = mpsut.npmps_to_uniten (psi5)
    #psi5,ens5,terrs5 = dmrg.dmrg (psi5, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1,psi2,psi3,psi4], weights=[20,20,20,20,20])
    #np.savetxt('terr5.dat',(maxdims,terrs5,ens5))

    #print(ens0[-1], ens1[-1], ens2[-1], ens3[-1], ens4[-1], ens5[-1])
    print(ens0[-1], ens1[-1])
    energy_1s= ens0[-1]
    energy_2s = ens1[-1]
    #exit()

    phi0 = mpsut.to_npMPS (psi0)
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
    with open(f'3d_dmrg_results_N={N}.pkl', 'wb') as f:
        pickle.dump(data, f)
