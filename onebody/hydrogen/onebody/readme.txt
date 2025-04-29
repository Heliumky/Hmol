I take imaginary time method, DMRG, discrete the exact(analytical form) solution and compare with the energy and density function as I plot.
################################
N_1d = 11
filename:1d_dmrg_density_functions.pdf
         1d_dftimt_density_functions.pdf : "dftimg" means discrete fourier transform imaginary time
         1d_ext_density_functions.pdf : "ext" means discrete the exact solution

################################


################################
for 2d case:
N_2d = 9
I try the different bond dimension and different steps:

filename:

2d_dmrg_density_functions_morestep.pdf : maxdims = [2]*10 + [4]*100 + [8]*200 + [16]*200 + [32]*100 

2d_dmrg_density_functions.pdf : maxdims = [2]*10 + [4]*100 + [8]*100 + [16]*100 + [32]*100

2d_dmrg_density_functions_incD.pdf : maxdims = [2]*10 + [4]*100 + [8]*100 + [16]*100 + [32]*100 + [64]*100

################################


Conclusion:
1.1D excited state can not get correct resulf from dmrg because the 1D psi_{2s} exist the analytical solution.

2.In 2d_dmrg_density_functions_morestep.pdf, we get the unsymertry density strange...

3.Use the plot_V, We can plot the 1/r potential for tuning parameters(scroll the parameters bar). N is number of site, shift is the boundary of real space. we use the TCI to fit this potential to avoid the cutoff.
