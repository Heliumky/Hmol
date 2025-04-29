import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 参数设定
rho_c = 2.0
N = 10000
h = rho_c / (N - 1)
rho = np.linspace(h, rho_c, N)  # 避开 rho = 0 的奇点
l = 0

# 势能项 U(rho) = (l^2 - 1/4)/rho^2 - 2/rho
U = (l**2 - 0.25) / rho**2 - 2 / rho

def numerov(E, U, rho, h):
    """使用 Numerov 方法从内向外积分，返回波函数"""
    f = U - 2 * E
    P = np.zeros_like(rho)
    P[0] = 0.0
    P[1] = 1e-5  # 初始小值
    for i in range(1, len(rho) - 1):
        P[i+1] = (2 * P[i] * (1 - (5*h**2/12)*f[i]) - P[i-1] * (1 + h**2/12*f[i-1])) / (1 + h**2/12*f[i+1])
    return P

def shoot(E_guess, node_target):
    """尝试能量 E_guess，计算节点数"""
    P = numerov(E_guess, U, rho, h)
    nodes = np.sum((P[:-1]*P[1:] < 0))  # 节点数
    return nodes, P

def find_state_energy(n_target, E_range, tol=1e-10):
    """用二分法找使节点数为 n_target 的能量"""
    E_low, E_high = E_range
    while E_high - E_low > tol:
        E_mid = (E_low + E_high) / 2
        nodes, _ = shoot(E_mid, n_target)
        if nodes > n_target:
            E_high = E_mid
        else:
            E_low = E_mid
    _, P_final = shoot((E_low + E_high) / 2, n_target)
    return (E_low + E_high) / 2, P_final

# === 找基态（n = 1，节点数 0） ===
E1, P1 = find_state_energy(0, [-2.0, 0.0])
print(f"Ground state (1s) energy ≈ {E1:.12f}")

# === 找第一个激发态（n = 2，节点数 1） ===
E2, P2 = find_state_energy(1, [-2.0, 2.0])
print(f"First excited state (2s) energy ≈ {E2:.12f}")

# 归一化并绘图
def normalize(P, rho):
    norm = np.sqrt(np.trapz(P**2, rho))
    return P / norm

P1n = normalize(P1, rho)
P2n = normalize(P2, rho)

plt.plot(rho, P1n, label="1s")
plt.plot(rho, P2n, label="2s")
plt.xlabel("ρ")
plt.ylabel("Radial wavefunction")
plt.legend()
plt.title("2D Hydrogen Atom (Confined at ρc=2.0)")
plt.grid()
plt.show()

