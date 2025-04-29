import numpy as np
import matplotlib.pyplot as plt

# 1D grid
x = np.linspace(-10, 10, 2048)  
dx = x[1] - x[0]

r = np.sqrt(x**2)
dr = 10*np.abs(r[0]-r[1])
# Potential (1D)
V = -1 / np.abs(x)  # 1D potential, similar to the 2D version but with a single coordinate

# Ground state 1D hydrogen: psi = delta(r)
psi_ground = (1/np.pi)*dr / (r**2  + dr**2)

# Normalization
norm_ground = np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)
psi_ground /= norm_ground

# Total ground state energy
kinetic_ground = 0.5 * (np.gradient(psi_ground, dx)**2)
potential_ground = V * np.abs(psi_ground)**2
energy_ground = np.sum(kinetic_ground + potential_ground) * dx

# First excited state 1D hydrogen: psi = (|x| - 3/4) * exp(-2/3 * |x|)
psi_excited =  r * np.exp(-r)

# Normalization for excited state
norm_excited = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)
psi_excited /= norm_excited

# Excited state total energy
kinetic_excited = 0.5 * (np.gradient(psi_excited, dx)**2)
potential_excited = V * np.abs(psi_excited)**2
energy_excited = np.sum(kinetic_excited + potential_excited) * dx

# Save wavefunctions to text files
ground_data = np.column_stack((x, np.real(psi_ground), np.imag(psi_ground)))
excited_data = np.column_stack((x, np.real(psi_excited), np.imag(psi_excited)))

#np.savetxt("1d_ext_ground_state_wavefunction.txt", ground_data, header="x Re(psi) Im(psi)", comments="")
#np.savetxt("1d_ext_excited_state_wavefunction.txt", excited_data, header="x Re(psi) Im(psi)", comments="")

# Save density plots as PDF
plt.figure(figsize=(12, 6))

# Ground state density
plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi_ground)**2, label=f'Energy = {energy_ground:.8f}')
plt.title('Ground State Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Excited state density
plt.subplot(1, 2, 2)
plt.plot(x, np.abs(psi_excited)**2, label=f'Energy = {energy_excited:.8f}', color='r')
plt.title('First Excited State Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig("1d_ext_density_functions.pdf", format='pdf')
plt.show()

print("Wavefunctions saved as '1d_ext_ground_state_wavefunction.txt' and '1d_ext_excited_state_wavefunction.txt'.")
print("Density functions saved as '1d_ext_density_functions.pdf'.")

