import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

m = 1.0     # mass
omega = 1.0 
k = 1     # coupling constant

M = 2 * m                   # Central mass
mu = m / 2                  # reduced mass
omega_bar = np.sqrt(omega**2 + 2 * k / m)  # 相对坐标的有效频率

# mesh grid
x = np.linspace(-10, 10, 512)  
dx = x[1] - x[0]

# central x and relative x
X, rel_x = np.meshgrid(x, x)

# potential
V_X = 0.5 * M * omega**2 * X**2
V_rel_x = 0.5 * mu * omega_bar**2 * rel_x**2

# **build up gs**
def harmonic_oscillator_wavefunction(n, x, mass, freq):
    """ n order hermite poly """
    H_n = hermite(n)  
    norm = 1.0 / np.sqrt(2**n * factorial(n))  # normalization
    alpha = np.sqrt(mass * freq)  # scalling factor in hermite 
    return norm * H_n(alpha * x) * np.exp(-0.5 * (alpha * x)**2)

# **基态波函数**
psi_X_ground = harmonic_oscillator_wavefunction(0, X, M, omega)
psi_rel_x_ground = harmonic_oscillator_wavefunction(1, rel_x, mu, omega_bar)
psi_ground = psi_X_ground * psi_rel_x_ground
print()

# **第一激发态波函数**
psi_X_excited = harmonic_oscillator_wavefunction(1, X, M, omega)
psi_rel_x_excited = harmonic_oscillator_wavefunction(1, rel_x, mu, omega_bar)
psi_excited = psi_X_excited * psi_rel_x_excited

# 基态波函数归一化
norm_ground = np.sum(np.abs(psi_ground)**2) * dx**2
psi_ground /= np.sqrt(norm_ground)

norm_excited = np.sum(np.abs(psi_excited)**2) * dx**2
psi_excited /= np.sqrt(norm_excited)

# 基态动能及总能量
#laplacian_groundX = np.gradient(np.gradient(psi_ground, dx, axis=0), dx, axis=0)
#laplacian_groundx_rel = np.gradient(np.gradient(psi_ground, dx, axis=1), dx, axis=1)

laplacian_groundX = np.gradient(psi_ground, dx, axis=0)**2
laplacian_groundx_rel = np.gradient(psi_ground, dx, axis=1)**2

kinetic_ground =  0.5/M * laplacian_groundX + 0.5/mu * laplacian_groundx_rel

#kinetic_ground = -0.5/M * np.conj(psi_ground) * laplacian_groundX - 0.5/mu * np.conj(psi_ground) * laplacian_groundx_rel
energy_ground = np.sum((kinetic_ground + (V_X + V_rel_x) * np.abs(psi_ground)**2)) * dx**2
print( np.sum(kinetic_ground )* dx**2)
print(np.sum((V_X + V_rel_x) * np.abs(psi_ground)**2) * dx**2)


# 第一激发态修复版
laplacian_excited_X = np.gradient(np.gradient(psi_excited, dx, axis=0), dx, axis=0)
laplacian_excited_rel = np.gradient(np.gradient(psi_excited, dx, axis=1), dx, axis=1)
laplacian_excited = laplacian_excited_X + laplacian_excited_rel
kinetic_excited = -0.5/M * np.conj(psi_excited) * laplacian_excited_X - 0.5/mu * np.conj(psi_excited) * laplacian_excited_rel
energy_excited = np.sum((kinetic_excited + (V_X + V_rel_x) * np.abs(psi_excited)**2)) * dx**2

# **绘制概率密度**
plt.figure(figsize=(12, 6))

# **基态密度**
plt.subplot(1, 2, 1)
plt.contourf(X, rel_x, np.abs(psi_ground)**2, levels=50, cmap='inferno')
plt.colorbar(label="Density")
plt.title(f'Ground State Density (Energy = {energy_ground:.4f})')
plt.xlabel('X (Center of Mass)')
plt.ylabel('x (Relative Coordinate)')

# **激发态密度**
plt.subplot(1, 2, 2)
plt.contourf(X, rel_x, np.abs(psi_excited)**2, levels=50, cmap='inferno')
plt.colorbar(label="Density")
plt.title(f'First Excited State Density (Energy = {energy_excited:.4f})')
plt.xlabel('X (Center of Mass)')
plt.ylabel('x (Relative Coordinate)')

plt.tight_layout()
plt.savefig("coupled_oscillators_density.pdf", format='pdf')
plt.show()

print("Density functions saved as 'coupled_oscillators_density.pdf'.")

