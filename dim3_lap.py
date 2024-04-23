import numpy as np
import taichi as ti
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
ti.init(ti.cpu)
len = [10,5,5] #目前仅使用len[0]进行测试
dim = 3
dw = [0.005,0.1,0.1,0.1]# 将所有的时间步长设置为0.005，dw[1]=0.1：一维空间步长
a3 = 5
b3 = 7

phi_la_3d = ti.field(ti.f32, shape=(int(len[2]/dw[3]), int(len[1]/dw[2]), int(len[0]/dw[1])))

@ti.kernel
def lap_solver_dim3(a3: ti.f32, b3: ti.f32):
    for i, j, k in phi_la_3d:
        # 首先设置所有边界为 b3
        if i == 0 or i == phi_la_3d.shape[0] - 1 or j == 0 or j == phi_la_3d.shape[1] - 1 or k == 0 or k == phi_la_3d.shape[2] - 1:
            phi_la_3d[i, j, k] = b3
        # 然后单独设置上下边界为 a3
        if k == 0 or k == phi_la_3d.shape[2] - 1:
            phi_la_3d[i, j, k] = a3


 
@ti.kernel
def lap_solver_dim3_test():
    for i, j, k in phi_la_3d:
        phi_la_3d[i, j, k] = 1.0




@ti.kernel
def lap_function_3d(dx: ti.f32, dy: ti.f32, dz: ti.f32, len_x: ti.i32, len_y: ti.i32, len_z: ti.i32):
    for i, j, k in ti.ndrange((1, len_x - 1), (1, len_y - 1), (1, len_z - 1)):
        phi_la_3d[i, j, k] = (phi_la_3d[i-1, j, k] + phi_la_3d[i+1, j, k] + phi_la_3d[i, j-1, k] + phi_la_3d[i, j+1, k] + phi_la_3d[i, j, k-1] + phi_la_3d[i, j, k+1]) / 6


def lap_convergence_3d():
    iteration_count = 0
    max_iterations = 100  # 限制最大迭代次数以避免无限循环
    while True: 
        phi_pre = phi_la_3d.to_numpy()
        lap_function_3d(dw[1], dw[2], dw[3], int(len[0]/dw[1]), int(len[1]/dw[2]), int(len[2]/dw[3]))
        phi_next = phi_la_3d.to_numpy()
        max_change = np.linalg.norm(phi_next - phi_pre, np.inf)
        print(f"Iteration {iteration_count}, Max change: {max_change}")
        if max_change < 1e-5 or iteration_count >= max_iterations:
            break
        iteration_count += 1



def print_field_3d():
    for k in range(int(len[2]/dw[3])):
        print(f"Slice at z[{k}]:")
        for j in range(int(len[1]/dw[2])):
            for i in range(int(len[0]/dw[1])):
                print(f"phi[{i}, {j}, {k}] = {phi_la_3d[i, j, k]}", end='  ')
            print()
        print("\n" + "-"*50)


if dim == 3:  
    lap_solver_dim3(a3, b3)
    print("hello")
    lap_convergence_3d()
    print_field_3d()