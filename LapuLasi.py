import numpy as np
import taichi as ti
import sympy as sp


ti.init(arch=ti.cpu)  # 初始化Taichi，使用CPU

# 参数定义
nx, ny, nz = 30, 30, 30  # 网格大小
iterations = 100  # 迭代次数

# 定义场
phi = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# 初始化边界条件
@ti.kernel
def init_boundary_conditions(a: ti.f32, b: ti.f32):
    for i, j, k in phi:
        # 根据输入的a和b设置边界条件
        if i == 0:
            phi[i, j, k] = a
        elif i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1:
            phi[i, j, k] = b

# 求解拉普拉斯方程
@ti.kernel
def solve_laplace():
    for _ in range(iterations):
        for i, j, k in ti.ndrange((1, nx-1), (1, ny-1), (1, nz-1)):
            phi[i, j, k] = (phi[i+1, j, k] + phi[i-1, j, k] +
                            phi[i, j+1, k] + phi[i, j-1, k] +
                            phi[i, j, k+1] + phi[i, j, k-1]) / 6.0

# 打印三维场
def print_field_3d():
    phi_np = phi.to_numpy()  # 将Taichi场转换为NumPy数组
    for k in range(nz):
        print(f"Slice at z[{k}]:")
        for j in range(ny):
            for i in range(nx):
                print(f"phi[{i}, {j}, {k}] = {phi_np[i, j, k]:.4f}", end='  ')
            print()
        print("\n" + "-"*50)

# 主程序
def main(a: float, b: float):
    init_boundary_conditions(a, b)  # 应用边界条件
    solve_laplace()  # 求解
    print_field_3d()  # 打印三维场的值

if __name__ == "__main__":
    a = 100.0  # 边界条件，可以根据需要进行调整
    b = 0.0
    main(a, b)





