import numpy as np
import scipy.linalg as la

# 参数设定
m = 1       # 质量
omega = 1  # 本征频率
k = 1       # 耦合常数

# 计算 M, mu, 和  \tilde{\omega}
M = 2 * m
mu = m / 2
omega_tilde = np.sqrt(omega**2 + 2 * k / m)

# 定义空间离散化
L = 5  # 空间范围
N = 100  # 格点数
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# 构造二阶微分矩阵 (-d^2/dx^2)
D2 = (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / dx**2

# Coupled Hamiltonian
H_coupled = np.kron(D2 / (2 * m), np.eye(N)) + np.kron(np.eye(N), D2 / (2 * m)) 
H_coupled += np.kron(np.diag(0.5 * m * omega**2 * x**2), np.eye(N))
H_coupled += np.kron(np.eye(N), np.diag(0.5 * m * omega**2 * x**2))
H_coupled += np.kron(np.eye(N), np.diag(0.5 * k * x**2))

# Decoupled Hamiltonian
H_decoupled = np.kron(D2 / (2 * M), np.eye(N)) + np.kron(np.eye(N), D2 / (2 * mu))
H_decoupled += np.kron(np.diag(0.5 * M * omega**2 * x**2), np.eye(N))
H_decoupled += np.kron(np.eye(N), np.diag(0.5 * mu * omega_tilde**2 * x**2))

# 计算基态能量
E_coupled = la.eigh(H_coupled, eigvals_only=True)[0]
E_decoupled = la.eigh(H_decoupled, eigvals_only=True)[0]

print(f"基态能量 (Coupled): {E_coupled}")
print(f"基态能量 (Decoupled): {E_decoupled}")

