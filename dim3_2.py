import numpy as np
import taichi as ti
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
ti.init(ti.cpu)
n = 50
dw = [0.005,0.1,0.1,0.1]
len = [10,5,5]


phi = ti.field(dtype=ti.f32, shape=(n, n, n))


a3, b3 = 0.0, 1.0  # 边界条件值

@ti.kernel
def init_phi():
    for i, j, k in phi:
        if i == 0 or i == n-1:
            phi[i, j, k] = a3 if i == 0 else b3
        else:
            phi[i, j, k] = 0

@ti.kernel
def update_phi(dx: ti.f32, dy: ti.f32, dz: ti.f32, len_x: ti.i32, len_y: ti.i32, len_z: ti.i32):
    for i, j, k in ti.ndrange((1, n-1), (1, n-1), (1, n-1)):
        phi[i, j, k] = (phi[i+1, j, k] + phi[i-1, j, k] +
                        phi[i, j+1, k] + phi[i, j-1, k] +
                        phi[i, j, k+1] + phi[i, j, k-1]) / 6.0

def iterate_until_convergence():
    while True: 
        phi_pre = phi.to_numpy()
        update_phi(dw[1], dw[2], dw[3], int(len[0]/dw[1]), int(len[1]/dw[2]), int(len[2]/dw[3]))
        phi_next = phi.to_numpy()
        if np.linalg.norm(phi_next - phi_pre, np.inf) < 1e-5:
            break

def main():
    init_phi()
    iterate_until_convergence()
    # 可以添加代码来可视化phi字段，或者进一步分析解

if __name__ == "__main__":
    main()
