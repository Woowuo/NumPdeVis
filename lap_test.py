import numpy as np
import taichi as ti
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ti.init(ti.cpu)
def str_to_function(expr_str,dim,dw,len1 = None,len2 = None, len3 = None):
    
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')
    # 将字符串表达式转换为sympy表达式
    expr = sp.sympify(expr_str)
    if dim == 1:
        f = []
        for i in range(0,int(len1/dw[1])):
            f.append(expr.subs({x:i*dw[1]}).evalf())
        print(f)
        return f
    if dim == 2:
        f = []
        for j in range(int(len2/dw[2])):
            row = []
            for i in range(int(len1/dw[1])):
                row.append(expr.subs({x: i*dw[1], y: j*dw[2]}).evalf())
            f.append(row)
        return f
        
    if dim == 3:
        f = []
        for k in range(int(len3/dw[3])):
            plane = []
            for j in range(int(len2/dw[2])):
                row = []
                for i in range(int(len1/dw[1])):
                    value = expr.subs({x: i*dw[1], y: j*dw[2], z: k*dw[3]}).evalf()
                    row.append(value)
                plane.append(row)
            f.append(plane)
        return f

len = [10,5,5] #目前仅使用len[0]进行测试
dim = 3
dw = [0.005,0.1,0.1,0.1]# 将所有的时间步长设置为0.005，dw[1]=0.1：一维空间步长
a1 = 0
b1 = 1
a2 = 3
b2 = 5
a3 = 0
b3 = 100
iterations = 100
nx, ny, nz = 50, 50, 50

phi_la = ti.field(ti.f32, shape=int(len[0]/dw[1]))
phi_la_2d = ti.field(ti.f32, shape=(int(len[1]/dw[2]), int(len[0]/dw[1])))
phi_la_3d = ti.field(ti.f32, shape=(int(len[2]/dw[3]), int(len[1]/dw[2]), int(len[0]/dw[1])))


# dim = 1, 一维
@ti.kernel
def lap_solver_dim1(a1: ti.f32, b1: ti.f32, dx: ti.f32, len: ti.f32):
    # 初始化边界条件 
    phi_la[0] = a1  # 左边界
    phi_la[int(len/dx) - 1] = b1  # 右边界

@ti.kernel
def lap_function(dx: ti.f32, len: ti.f32):
    for i in ti.ndrange((1, int(len / dx) - 1)):
        phi_la[i] = 0.5 * (phi_la[i+1] + phi_la[i-1])  

def lap_convergence():
    # 测试迭代直到收敛后停止
    while 1: 
        phi_pre = phi_la.to_numpy()
        lap_function(dw[1],len[0])
        phi_next = phi_la.to_numpy()
        if np.linalg.norm(phi_next - phi_pre, np.inf) < 1e-5:
            break
 
@ti.kernel  # dim=2，二维
def lap_solver_dim2(a2: ti.f32, b2: ti.f32, dx: ti.f32, dy: ti.f32, len_x: ti.i32, len_y: ti.i32):
    for i, j in ti.ndrange(len_x, len_y):
        if i == 0 or i == len_x - 1:  # 左侧和右侧边界
            phi_la_2d[i, j] = b2
        if j == 0 or j == len_y - 1:  # 顶部和底部边界
            phi_la_2d[i, j] = a2
  

@ti.kernel
def lap_function_2d(dx: ti.f32, dy: ti.f32, len_x: ti.i32, len_y: ti.i32):
    for i, j in ti.ndrange((1, len_x - 1), (1, len_y - 1)):
        phi_la_2d[i, j] = 0.25 * (phi_la_2d[i-1, j] + phi_la_2d[i+1, j] + phi_la_2d[i, j-1] + phi_la_2d[i, j+1])


def lap_convergence_2d():
    while True: 
        phi_pre = phi_la_2d.to_numpy()
        lap_function_2d(dw[1], dw[2], int(len[0]/dw[1]), int(len[1]/dw[2]))
        phi_next = phi_la_2d.to_numpy()
        if np.linalg.norm(phi_next - phi_pre, np.inf) < 1e-5:
            break



# dim = 3, 三维
@ti.kernel
def lap_solver_dim3(a3: ti.f32, b3: ti.f32, dx: ti.f32, dy: ti.f32, dz: ti.f32, len_x: ti.i32, len_y: ti.i32, len_z: ti.i32):
    for i, j, k in ti.ndrange(len_x, len_y, len_z):
        if i == 0 or i == len_x - 1 or j == 0 or j == len_y - 1 or k == 0 or k == len_z - 1:  # 上下边界
            phi_la_3d[i, j, k] = a3       
        if i == 0:   # 其余边界
            phi_la_3d[i, j, k] = b3

@ti.kernel
def init_boundary_conditions():
    for i, j, k in phi_la_3d:
        # 假设所有边界上的值为0，除了x=0处为100
        if i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1:
            phi_la_3d[i, j, k] = 0.0
        if i == 0:
            phi_la_3d[i, j, k] = 100.0

@ti.kernel
def lap_function_3d(dx: ti.f32, dy: ti.f32, dz: ti.f32, len_x: ti.i32, len_y: ti.i32, len_z: ti.i32):
    for i, j, k in ti.ndrange((1, len_x - 1), (1, len_y - 1), (1, len_z - 1)):
        phi_la_3d[i, j, k] = (phi_la_3d[i-1, j, k] + phi_la_3d[i+1, j, k] + phi_la_3d[i, j-1, k] + phi_la_3d[i, j+1, k] + phi_la_3d[i, j, k-1] + phi_la_3d[i, j, k+1]) / 6

@ti.kernel
def solve_laplace_3d():
    for _ in range(iterations):
        for i, j, k in ti.ndrange((1, nx-1), (1, ny-1), (1, nz-1)):
            phi_la_3d[i, j, k] = (phi_la_3d[i+1, j, k] + phi_la_3d[i-1, j, k] +
                            phi_la_3d[i, j+1, k] + phi_la_3d[i, j-1, k] +
                            phi_la_3d[i, j, k+1] + phi_la_3d[i, j, k-1]) / 6.0

def lap_convergence_3d():
    while True: 
        phi_pre = phi_la_3d.to_numpy()
        lap_function_3d(dw[1], dw[2], dw[3], int(len[0]/dw[1]), int(len[1]/dw[2]), int(len[2]/dw[3]))
        phi_next = phi_la_3d.to_numpy()
        if np.linalg.norm(phi_next - phi_pre, np.inf) < 1e-5:
            break


@ti.kernel
def print_mid_slice_3d():
    nx = int(len[2]/dw[3])
    ny = int(len[1]/dw[2])
    nz = int(len[0]/dw[1])
    for i, j in ti.ndrange(( 0, nx), (0, ny) ):
        print(phi_la_3d[i, j, nz // 2], end=' ')
    print()

# 打印函数
def print_field():
    for i in range(int(len[0]/dw[1])):
        print(f"x[{i}] = {phi_la.to_numpy()[i]}")

def print_field_2d():
    for j in range(int(len[1]/dw[2])):
        for i in range(int(len[0]/dw[1])):
            print(f"phi[{i}, {j}] = {phi_la_2d[i, j]:.4f}", end='  ')
        print()

def print_field_3d():
    for k in range(int(len[2]/dw[3])):
        print(f"Slice at z[{k}]:")
        for j in range(int(len[1]/dw[2])):
            for i in range(int(len[0]/dw[1])):
                print(f"phi[{i}, {j}, {k}] = {phi_la_3d[i, j, k]:.4f}", end='  ')
            print()
        print("\n" + "-"*50)




if dim == 1:    
    lap_solver_dim1(a1,b1,dw[1],len[0])
    lap_convergence()
    print_field()
elif dim == 2:
    lap_solver_dim2(a2, b2, dw[1], dw[2], int(len[0]/dw[1]), int(len[1]/dw[2]))
    lap_convergence_2d()
    print_field_2d()
elif dim == 3:  
    # lap_solver_dim3(a3, b3, dw[1], dw[2], dw[3], int(len[0]/dw[1]), int(len[1]/dw[2]), int(len[2]/dw[3]))
    init_boundary_conditions()
    # lap_convergence_3d()
    solve_laplace_3d()
    print_mid_slice_3d()
    # print_field_3d()
