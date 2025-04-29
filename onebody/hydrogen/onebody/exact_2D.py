import numpy as np
import matplotlib.pyplot as plt

# meshgrid
N = 9
shift = -5

x = np.linspace(shift, -shift, 2**N)
y = np.linspace(shift, -shift, 2**N)
X, Y = np.meshgrid(x, y)  

# potential
V = -1 / np.sqrt(X**2 + Y**2) 
#V = 0.5*(X**2 + Y**2)
# polar coordinate
r = np.sqrt(X**2 + Y**2)

plt.figure(figsize=(12, 6))
plt.plot(x, -1 / np.sqrt(x**2))

# ground state 2D hydrogen：psi = exp(-2r)
psi_ground = np.exp(-2 * r)
dx = x[1] - x[0]
dy = y[1] - y[0]
print(dx)

def second_derivative_nd(f, dx, axis):
    return (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / dx**2

# normalization
norm_ground = np.sqrt(np.sum(np.abs(psi_ground)**2) * dx * dy)
psi_ground /= norm_ground

# total ground state energy
#kinetic_ground = 0.5 * (np.abs(np.gradient(psi_ground, dx, axis=0))**2 + np.abs(np.gradient(psi_ground, dy, axis=1))**2)

kinetic_ground = -0.5*(np.conj(psi_ground) * second_derivative_nd(psi_ground, dx, axis=0) + np.conj(psi_ground) * second_derivative_nd(psi_ground, dy, axis=1))


potential_ground = V * np.abs(psi_ground)**2
energy_ground = np.sum(kinetic_ground + potential_ground) * dx * dy

#2D hydrogen first excited state：psi = (r - 3/4) * exp(-2/3 * r)
psi_excited = (r - 3/4) * np.exp(-2/3 * r)

# normalization fs
norm_excited = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx * dy)
psi_excited /= norm_excited

# fs total energy
#kinetic_excited = 0.5 * (np.abs(np.gradient(psi_excited, dx, axis=0))**2 + np.abs(np.gradient(psi_excited, dy, axis=1))**2)

kinetic_excited = -0.5*(np.conj(psi_excited) * second_derivative_nd(psi_excited, dx, axis=0) + np.conj(psi_excited) * second_derivative_nd(psi_excited, dy, axis=1))

potential_excited = V * np.abs(psi_excited)**2
energy_excited = np.sum(kinetic_excited + potential_excited) * dx * dy

import numpy as np
import matplotlib.pyplot as plt

# Save wavefunctions to text files
ground_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_ground).flatten(), np.imag(psi_ground).flatten()))
excited_data = np.column_stack((X.flatten(), Y.flatten(), np.real(psi_excited).flatten(), np.imag(psi_excited).flatten()))

np.savetxt("2d_ext_ground_state_wavefunction.txt", ground_data, header="x y Re(psi) Im(psi)", comments="")
np.savetxt("2d_ext_excited_state_wavefunction.txt", excited_data, header="x y Re(psi) Im(psi)", comments="")

# Save density plots as PDF
plt.figure(figsize=(12, 6))

# Ground state density
plt.subplot(1, 2, 1)
plt.imshow(np.abs(psi_ground)**2, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.title(f'Ground State Density (Energy = {energy_ground:.8f})')
plt.xlabel('x')
plt.ylabel('y')

# Excited state density
plt.subplot(1, 2, 2)
plt.imshow(np.abs(psi_excited)**2, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='plasma')
plt.colorbar(label='Intensity')
plt.title(f'First Excited State Density (Energy = {energy_excited:.8f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig(f"2d_ext_density_functions{N}.pdf", format='pdf')
plt.show()

print("Wavefunctions saved as '2d_ext_ground_state_wavefunction.txt' and '2d_ext_excited_state_wavefunction.txt'.")
print("Density functions saved as '2d_ext_density_functions.pdf'.")

