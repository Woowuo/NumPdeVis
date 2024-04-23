import taichi as ti
import sympy as sp
import latex2sympy2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@ti.data_oriented
class laplace_equ():
    def __init__(self,len,dim,dw,a,b):
        self.len = len
        self.dim = dim
        self.dw = dw
        self.a = a
        self.b = b
        # dim1
        self.phi_x = ti.field(ti.f32, shape=int(self.len[0] / self.dw[0]))
        # dim2
        self.phi_xy = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[0]), int(self.len[1] / self.dw[1])))
        # dim3
        self.phi_xyz = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[0]), int(self.len[1] / self.dw[1]), int(self.len[2] / self.dw[2])))

    @ti.kernel
    def laplace_solver_dim1(self):
        # 初始化边界条件
        a1 = self.a[0]
        b1 = self.b[0]
        dx = self.dw[0]
        len = self.len[0]
        n = int(len / dx)  # 网格点数量

        # 线性插值初始化内部点
        self.phi_x[0]= a1
        self.phi_x[int(len / dx)-1] = b1
        for i in range(1, n - 1):
            self.phi_x[i] = a1 + (b1 - a1) * i / (n - 1)

        # 标记是否收敛，这里用一个较大的数开始
        max_diff = 1.0
        while max_diff > 1e-5:  # 收敛条件
            max_diff = 0.0
            for i in range(1, n - 1):
                old_value = self.phi_x[i]
                # 使用拉普拉斯方程的离散近似更新内部点
                self.phi_x[i] = 0.5 * (self.phi_x[i - 1] + self.phi_x[i + 1])
                # 更新最大差异
                diff = abs(self.phi_x[i] - old_value)
                if diff > max_diff:
                    max_diff = diff

    @ti.kernel
    def laplace_solver_dim2(self):
        a1, a2 = self.a[0], self.a[1]  # 沿x和y的边界条件
        b1, b2 = self.b[0], self.b[1]  # 沿x和y的边界条件
        dx = self.dw[1]
        dy = self.dw[2]
        len1, len2 = self.len[0], self.len[1]
        n_x = int(len1 / dx)
        n_y = int(len2 / dy)

        # 边界条件初始化
        for j in ti.ndrange((0, n_y)):  # x方向的边界
            self.phi_xy[0, j] = a1
            self.phi_xy[n_x - 1, j] = b1
        for i in ti.ndrange((0, n_x)):  # y方向的边界
            self.phi_xy[i, 0] = a2
            self.phi_xy[i, n_y - 1] = b2

        # 线性插值初始化内部点
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                # 插值
                self.phi_xy[i, j] = (a1 * (n_x - 1 - i) + b1 * i) / (n_x - 1) + (a2 * (n_y - 1 - j) + b2 * j) / (
                            n_y - 1)

        # 迭代更新内部点，并检测收敛
        max_diff = 1.0  # 初始设为较大值以进入循环
        while max_diff > 1e-5:  # 收敛阈值
            max_diff = 0.0
            for i in range(1, n_x - 1):
                for j in range(1, n_y - 1):
                    old_value = self.phi_xy[i, j]
                    # 根据拉普拉斯方程的离散近似更新内部点
                    self.phi_xy[i, j] = 0.25 * (
                                self.phi_xy[i + 1, j] + self.phi_xy[i - 1, j] + self.phi_xy[i, j + 1] + self.phi_xy[
                            i, j - 1])
                    # 计算更新后的最大差异
                    diff = abs(self.phi_xy[i, j] - old_value)
                    if diff > max_diff:
                        max_diff = diff

    @ti.kernel
    def laplace_solver_dim3(self):
        # 定义边界条件
        a1, a2, a3 = self.a[0], self.a[1], self.a[2]
        b1, b2, b3 = self.b[0], self.b[1], self.b[2]
        dx = self.dw[0]
        dy = self.dw[1]
        dz = self.dw[2]
        len1, len2, len3 = self.len[0], self.len[1], self.len[2]
        n_x = int(len1 / dx)
        n_y = int(len2 / dy)
        n_z = int(len3 / dz)
        for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
            if i == 0 or i == int(len1 / dx) - 1:
                self.phi_xyz[i, j, k] = a1 if i == 0 else b1
            if j == 0 or j == int(len2 / dy) - 1:
                self.phi_xyz[i, j, k] = a2 if j == 0 else b2
            if k == 0 or k == int(len3 / dz) - 1:
                self.phi_xyz[i, j, k] = a3 if k == 0 else b3
        for i, j, k in ti.ndrange((0, n_x), (0, n_y), (0, n_z)):
            if not (i == 0 or i == n_x - 1 or j == 0 or j == n_y - 1 or k == 0 or k == n_z - 1):
                # 对于内部点，进行线性插值
                # 以x方向为例，其他方向类似
                interp_x = a1 + (b1 - a1) * i / (n_x - 1)
                interp_y = a2 + (b2 - a2) * j / (n_y - 1)
                interp_z = a3 + (b3 - a3) * k / (n_z - 1)
                # 将三个方向的插值结果进行平均，作为该点的初始估计值
                self.phi_xyz[i, j, k] = (interp_x + interp_y + interp_z) / 3
        max_diff = 1.0  # 初始设为较大值以进入循环
        while max_diff > 1e-5:  # 收敛阈值
            max_diff = 0.0
            for i, j, k in ti.ndrange((1, n_x - 1), (1, n_y - 1), (1, n_z - 1)):
                old_value = self.phi_xyz[i, j, k]
                # 根据拉普拉斯方程的离散近似更新内部点
                self.phi_xyz[i, j, k] = (self.phi_xyz[i + 1, j, k] + self.phi_xyz[i - 1, j, k] +
                                         self.phi_xyz[i, j + 1, k] + self.phi_xyz[i, j - 1, k] +
                                         self.phi_xyz[i, j, k + 1] + self.phi_xyz[i, j, k - 1]) / 6.0
                # 计算更新后的最大差异
                diff = abs(self.phi_xyz[i, j, k] - old_value)
                if diff > max_diff:
                    max_diff = diff

    def print_field_dim1(self):
        for i in range(0, int(self.len[0] / self.dw[0])):
            print(f"x[{i}] = {self.phi_x[i]}")  # 打印field的每个元素

    def print_field_dim2(self):
        for i, j in ti.ndrange((0, int(self.len[0] / self.dw[0])), (0, int(self.len[1] / self.dw[1]))):
            print(f"x[{i}, {j}] = {self.phi_xy[i, j]}")  # 打印field的每个元素

    def print_field_dim3(self):
        for i, j, k in ti.ndrange((0, int(self.len[0] / self.dw[0])), (0, int(self.len[1] / self.dw[1])),
                                     (0, int(self.len[2] / self.dw[2]))):
            print(f"x[{i}, {j},{k}] = {self.phi_xyz[i, j, k]}")  # 打印field的每个元素

    def draw_dim1(self):
        y = self.phi_x.to_numpy()
        dx = self.dw[0]
        len_x = self.len[0]
        x = np.arange(0, len_x, dx)

        # 绘图
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('laplace equation')
        plt.show()

    def draw_dim2(self):
        Z = self.phi_xy.to_numpy()
        dx = self.dw[0]
        dy = self.dw[1]
        len1 = self.len[0]
        len2 = self.len[1]
        x = np.arange(0, len1, dx)
        y = np.arange(0, len2, dy)
        X, Y = np.meshgrid(y, x)

        # 创建图形和轴
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')

        # 添加颜色栏
        fig.colorbar(surf)

        # 显示图形
        plt.show()

    def exe_dim1_solver(self):
        self.laplace_solver_dim1()
        self.draw_dim1()

    def exe_dim2_solver(self):
        self.laplace_solver_dim2()
        self.draw_dim2()

    def exe_dim3_solver(self):
        self.laplace_solver_dim3()
        self.print_field_dim3()



if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    # a = laplace_equ([5,5,5],1,[0.1,0.1,0.1],[4,0,0],[0,0,0])
    # a.exe_dim1_solver()
    a = laplace_equ([10,10,10],2,[0.1,0.1,0.1],[0,0,0],[5,5,5])
    a.exe_dim2_solver()
    # a = laplace_equ([10,5,5],1,[0.1,0.1,0.1],[5,10,0],[5,0,5])
    # a.exe_dim3_solver()