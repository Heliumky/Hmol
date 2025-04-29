import numpy as np
import matplotlib.pyplot as plt

# 1D meshgrid
x = np.linspace(-10, 10, 2048)
dx = x[1] - x[0]

# SHO and 1D Hydrogen potential
V = -1 / np.abs(x)  # 1D Hydrogen potential
# V = 0.5 * x**2    # 1D SHO potential

# Initial wavefunction (random complex function)
np.random.seed(42)
real_part = np.random.rand(len(x)) - 0.5
imag_part = np.random.rand(len(x)) - 0.5
psi = (real_part + 1j * imag_part).astype(np.complex128)

# Normalization
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
psi /= norm

# Build up k-space
kx = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
K2 = kx**2

# Imaginary time evolution parameters
tau = 0.001
max_steps_gd = 10000
max_steps_fs = 40000

# Ground state computation
psi_k = np.fft.fft(psi) * np.exp(K2 / 4 * tau)
for step in range(max_steps_gd):
    psi = np.fft.ifft(psi_k)
    psi *= np.exp(-V * tau)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi /= norm
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-K2 * tau / 2)

psi_k *= np.exp(K2 * tau / 4)
psi_ground = np.fft.ifft(psi_k)
psi_ground /= np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)

# Ground state energy
energy_ground = np.sum(
    0.5 * (np.abs(np.gradient(psi_ground, dx))**2) + V * np.abs(psi_ground)**2
) * dx

# Initialize first-excited state
np.random.seed(43)
real_part = np.random.rand(len(x)) - 0.5
imag_part = np.random.rand(len(x)) - 0.5
psi_excited = (real_part + 1j * imag_part).astype(np.complex128)

# Normalization
norm = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)
psi_excited /= norm

# Imaginary time evolution for the first excited state
psi_k = np.fft.fft(psi_excited) * np.exp(K2 / 4 * tau)
for step in range(max_steps_fs):
    psi_excited = np.fft.ifft(psi_k)
    psi_excited *= np.exp(-V * tau)

    # Make the excited state orthogonal to the ground state
    overlap = np.sum(np.conj(psi_ground) * psi_excited) * dx
    psi_excited -= overlap * psi_ground

    # Normalization
    norm = np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)
    psi_excited /= norm
    psi_k = np.fft.fft(psi_excited)
    psi_k *= np.exp(-K2 * tau / 2)

psi_k *= np.exp(K2 * tau / 4)
psi_excited = np.fft.ifft(psi_k)
psi_excited /= np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)

# First excited state energy
energy_excited = np.sum(
    0.5 * (np.abs(np.gradient(psi_excited, dx))**2) + V * np.abs(psi_excited)**2
) * dx

# save wf gs 
ground_data = np.column_stack((x, np.real(psi_ground), np.imag(psi_ground)))
np.savetxt("1d_dftimt_ground_state_wavefunction.txt", ground_data, header="x Re(psi) Im(psi)", comments="")

# save wf fs
excited_data = np.column_stack((x, np.real(psi_excited), np.imag(psi_excited)))
np.savetxt("1d_dftimt_excited_state_wavefunction.txt", excited_data, header="x Re(psi) Im(psi)", comments="")

# ploting PDF
plt.figure(figsize=(12, 6))

# gs
plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi_ground)**2, label=f"Energy = {energy_ground:.8f}")
plt.title("Ground State Density")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

# fs
plt.subplot(1, 2, 2)
plt.plot(x, np.abs(psi_excited)**2, label=f"Energy = {energy_excited:.8f}", color='r')
plt.title("First Excited State Density")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig("1d_dftimt_density_functions.pdf", format="pdf")
plt.show()

print("Wavefunctions saved as '1d_dftimt_ground_state_wavefunction.txt' and '1d_dftimt_excited_state_wavefunction.txt'.")
print("Density functions saved as '1d_dftimt_density_functions.pdf'.")
