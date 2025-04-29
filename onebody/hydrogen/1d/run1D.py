import dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut
import differential as df
import copy, sys, os
#sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
#sys.path.insert(0,'/home/chiamin/cytnx_new/')
import cytnx
import qn_utility as qn
import MPS_utility as mpsut
import plot_utility as ptut
from matplotlib import cm
import hamilt
from test import load_mps
import npmps

if __name__ == '__main__':
    N = 11
    #cutoff = 0.1
    shift = - 10
    rescale = -2*shift/(2**N-1)
    xmax = -shift
    cutoff = rescale
    dx = rescale 
    print('xmax =',xmax)
    print('xshift =',shift)
    print('dx =',dx)



    # Generate the MPS for the potential


    # Kinetic energy
    Hk, Lk, Rk = hamilt.H_kinetic(N, dx)
    
    # Potential energy
    factor = 1.
    os.system('python3 tci.py '+str(N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --1D_one_over_r')
    V_MPS = load_mps('fit.mps.npy')
    #V_MPS = tci.tci_one_over_r_2D (2*N, rescale, cutoff, factor, shift)
    HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    print(len(Hk),len(HV))
    H, L, R = npmps.sum_2MPO (Hk, Lk, Rk, HV, LV, RV)

    H, L, R = ut.compress_mpo (H, L, R, cutoff=1e-12)

    # Create MPS


    # -------------- DMRG -------------
    H, L, R = mpsut.npmpo_to_uniten (H, L, R)

    # Define the bond dimensions for the sweeps
    maxdims = [2]*10 + [4]*10 + [8]*80 + [16]*100 + [32]*100
    cutoff = 1e-12

    # Run dmrg
    psi0 = npmps.random_MPS (N, 2, 2)
    psi0 = mpsut.npmps_to_uniten (psi0)
    psi0, ens0, terrs0 = dmrg.dmrg (psi0, H, L, R, maxdims, cutoff)
    np.savetxt('terr0_1d.dat',(terrs0,ens0))

    maxdims = maxdims + [64]*80
    psi1 = npmps.random_MPS (N, 2, 2)
    psi1 = mpsut.npmps_to_uniten (psi1)
    psi1,ens1,terrs1 = dmrg.dmrg (psi1, H, L, R, maxdims, cutoff, ortho_mpss=[psi0], weights=[20])
    np.savetxt('terr1_1d.dat',(terrs1,ens1))

    maxdims = maxdims + [64]*256
    psi2 = npmps.random_MPS (N, 2, 2)
    psi2 = mpsut.npmps_to_uniten (psi2)
    psi2,ens2,terrs2 = dmrg.dmrg (psi2, H, L, R, maxdims, cutoff, ortho_mpss=[psi0,psi1], weights=[20,20])
    np.savetxt('terr2_2d.dat',(terrs2,ens2))

    print(ens0[-1], ens1[-1])#, ens2[-1])


    phi0 = mpsut.to_npMPS (psi0)
    phi1 = mpsut.to_npMPS (psi1)
    phi2 = mpsut.to_npMPS (psi2)

    energy_ground = ens0[-1]
    energy_fs_excited = ens1[-1]
    energy_ss_excited = ens2[-1]

    # --------------- Plot ---------------------
    bxs = list(ptut.BinaryNumbers(N))
    xs = ptut.bin_to_dec_list (bxs,rescale=rescale, shift=shift)


    # The potential
    Vx = [ptut.get_ele_mps (V_MPS, bx) for bx in bxs]

    ys0 = [ptut.get_ele_mps (phi0, bx) for bx in bxs]
    ys1 = [ptut.get_ele_mps (phi1, bx) for bx in bxs]
    ys2 = [ptut.get_ele_mps (phi2, bx) for bx in bxs]

    #fig, ax = plt.subplots()
    #ax.plot (xs, Vx)
    #ax.plot (xs, (np.array(ys0)/np.sqrt(dx))**2, c ='red', label = f'ground density')
    #ax.plot (xs, (np.array(ys1)/np.sqrt(dx))**2, c = 'blue', label = f'first excited stat')
    #ax.plot (xs, ys2, marker='.')
    #plt.legend()
    #plt.show()

    # Save density plots as PDF
    plt.figure(figsize=(12, 6))
    # Ground state density
    plt.subplot(1, 3, 1)
    plt.plot(xs, np.abs(np.array(ys0)/np.sqrt(dx))**2, label=f'Energy = {energy_ground:.8f}')
    plt.title('Ground State Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()

    # Excited state density
    plt.subplot(1, 3, 2)
    plt.plot(xs, np.abs(np.array(ys1)/np.sqrt(dx))**2, label=f'Energy = {energy_fs_excited:.8f}', color='r')
    plt.title('First Excited State Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()

    # Excited state density
    plt.subplot(1, 3, 3)
    plt.plot(xs, np.abs(np.array(ys2)/np.sqrt(dx))**2, label=f'Energy = {energy_ss_excited:.8f}', color='r')
    plt.title('Second Excited State Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()


    plt.tight_layout()
    plt.savefig("1d_dmrg_density_functions.pdf", format='pdf')
    plt.show()
