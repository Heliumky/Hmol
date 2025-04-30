import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
#import dmrg as dmrg
#import numpy as np
#import matplotlib.pyplot as plt
#import qtt_utility as ut
import copy, sys, os
#import MPS_utility as mpsut
#import twobody as twut
import SHO as sho
import plot_utility as ptut
from matplotlib import cm
import hamilt
import tci
#import npmps
#import plotsetting as ps

if __name__ == '__main__':
    N = 9
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
    #Hk, Lk, Rk = hamilt.H_kinetic(N, dx)
    #HV, LV, RV = hamilt.H_trap(N, dx, shift)
    # Potential energy
    #factor = 1.
    #os.system('python3 tci.py '+str(N)+' '+str(rescale)+' '+str(shift)+' '+str(cutoff)+' '+str(factor)+' --1D_one_over_r')
    # Load the potential MPS
    #V_MPS = load_mps('fit.mps.npy')
    #HV, LV, RV = npmps.mps_func_to_mpo(V_MPS)

    #print(len(Hk),len(HV))
    #H1, L1, R1 = npmps.sum_2MPO (Hk, 0.5*Lk, Rk, HV, 2*LV, RV)
    #H2, L2, R2 = npmps.sum_2MPO (Hk, 2*Lk, Rk, HV, 1.5*LV, RV)

    # Create a two-particle Hamiltonian
    #H, L, R = twut.H_two_particles (H1, L1, R1, H2, L2, R2)

    #H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)
    #H, L, R = npmps.change_dtype (H, L, R, dtype=complex)
    # Rotate H to the parity basis
    #U = twut.get_swap_U()
    #H = ut.applyLocalRot_mpo (H, U)

    # Set quantum numbers
    #H, L, R = twut.set_mpo_quantum_number (H, L, R)
    
    #H, L, R = mpsut.to_npMPO(H, L, R)

    #H, L, R = npmps.compress_MPO (H, L, R, cutoff=1e-12)

    #H = npmps.absort_LR (H, L, R)
    #print([H[i].shape for i in range(len(H))])
    Hmtx = npmps.MPO_to_matrix (H)

    import scipy as sp
    #print(Hmtx)
    #gsen, psi = sp.linalg.eigh(Hmtx,  subset_by_index=[0, 0])
    #print(eigenvalues[0])

    #psi = ut.applyLocal_mps (psi, U)

    #corr = twut.corr_matrix (psi)
    #Lcorr = np.array([1.])
    #Rcorr = np.array([1.])

    # Target the largest occupations
    #corr[0] *= -1.
    #corr, Lcorr, Rcorr = npmps.compress_MPO (corr, Lcorr, Rcorr, cutoff=1e-12)
    #corr, Lcorr, Rcorr = npmps.change_dtype (corr, Lcorr, Rcorr, dtype=complex)




