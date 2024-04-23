import numpy as np
import taichi as ti
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ti.init(ti.cpu)

class LaplaceSolver:
    def __init__(self, dim, length, dw, boundary_conditions):
        self.dim = dim  # 维度（1，2或3）
        self.length = length  # 长度
        self.dw = dw  # 宽度
        self.boundary_conditions = boundary_conditions  # 边界条件
        
       
        if dim == 1:
            self.phi_la = ti.field(ti.f32, shape=int(length[0]/dw[1]))
        elif dim == 2:
            self.phi_la_2d = ti.field(ti.f32, shape=(int(length[1]/dw[2]), int(length[0]/dw[1])))
        elif dim == 3:
            self.phi_la_3d = ti.field(ti.f32, shape=(int(length[2]/dw[3]), int(length[1]/dw[2]), int(length[0]/dw[1])))

    def str_to_function(self, expr_str, len1=None, len2=None, len3=None):
        # 实现将字符串表达式转换为函数的方法
        pass
    @ti.kernel
    def lap_solver_dim1(a1: ti.f32, b1: ti.f32, dx: ti.f32, len: ti.f32):
        # 1D
        pass
    @ti.kernel
    def lap_solver_dim2(self, a2: ti.f32, b2: ti.f32, dx: ti.f32, dy: ti.f32, len_x: ti.i32, len_y: ti.i32):
        # 2D
        pass
    @ti.kernel
    def lap_solver_dim3(self, a3: ti.f32, b3: ti.f32, dx: ti.f32, dy: ti.f32, dz: ti.f32, len_x: ti.i32, len_y: ti.i32, len_z: ti.i32):
        # 3D
        pass
    def lap_convergence(self):
        pass
    
    def print_field(self):
        # 打印1D
        pass
    def print_field_2d(self):
        # 打印2D
        pass
    def print_field_3d(self):
        # 打印3D
        pass

boundary_conditions = [(0, 1), (3, 5), (5, 7)]
length = [10, 5, 5]
dw = [0.005, 0.1, 0.1, 0.1]
solver = LaplaceSolver(dim=3, length=length, dw=dw, boundary_conditions=boundary_conditions)


