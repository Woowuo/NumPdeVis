import numpy as np
import taichi as ti
import sympy as sp
import latex2sympy2


def str_to_function(expr_str, dim, dw, len1=None, len2=None, len3=None):
    """
    将 LaTeX 字符串形式的数学表达式转换为 Sympy 表达式，并根据维度返回计算后的数值列表。
    """
    # 将 LaTeX 字符串转换为 Sympy 表达式
    expr = latex2sympy2.latex2sympy(expr_str)

    # 定义符号
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')

    if dim == 1:
        f = [expr.subs(x, i * dw[1]).evalf() for i in range(int(len1 / dw[1]))]
        return f
    elif dim == 2:
        f = [[expr.subs({x: i * dw[1], y: j * dw[2]}).evalf() for j in range(int(len2 / dw[2]))] for i in
             range(int(len1 / dw[1]))]
        return f
    elif dim == 3:
        total_iterations = int(len1 / dw[1]) * int(len2 / dw[2]) * int(len3 / dw[3])
        current_iteration = 0
        f = []
        for i in range(int(len1 / dw[1])):
            layer = []
            for j in range(int(len2 / dw[2])):
                row = []
                for k in range(int(len3 / dw[3])):
                    row.append(expr.subs({x: i * dw[1], y: j * dw[2], z: k * dw[3]}).evalf())
                    current_iteration += 1
                    if current_iteration % (total_iterations // 100) == 0:  # Update progress every 1%
                        print(
                            f"Progress: {current_iteration}/{total_iterations} ({current_iteration / total_iterations * 100:.2f}%)")
                layer.append(row)
            f.append(layer)
        return f


@ti.data_oriented
class height_equ():
    def __init__(self, func_string, len, dim, v, dw, a, b, t):
        self.func_string = func_string
        self.len = len
        self.dim = dim
        self.v = v
        self.dw = dw
        self.a = a
        self.b = b
        self.t = t
        # dim1
        self.phi_xt = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.t / self.dw[0])))
        self.func_dim1 = ti.field(ti.f32, shape=int(self.len[0] / self.dw[1]))
        # dim2
        self.phi_xyt = ti.field(ti.f32, shape=(
        int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]), int(self.t / self.dw[0])))
        self.func_dim2 = ti.field(ti.f32, shape=(int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2])))
        # dim3
        self.phi_xyzt = ti.field(ti.f32, shape=(
        int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]), int(self.len[2] / self.dw[3]),
        int(self.t / self.dw[0])))
        self.func_dim3 = ti.field(ti.f32, shape=(
        int(self.len[0] / self.dw[1]), int(self.len[1] / self.dw[2]), int(self.len[2] / self.dw[3])))

    def init_func(self):
        f1 = str_to_function(self.func_string, self.dim, self.dw, self.len[0], self.len[1], self.len[2])
        if self.dim == 1:
            for i in range(0, int(self.len[0] / self.dw[1])):
                # 确保赋值前将结果转换为float类型
                self.func_dim1[i] = f1[i]
        if self.dim == 2:
            for i in range(0, int(self.len[0] / self.dw[1])):
                for j in range(0, int(self.len[1] / self.dw[2])):
                    self.func_dim2[i, j] = f1[i][j]
        if self.dim == 3:
            for i in range(0, int(self.len[0] / self.dw[1])):
                for j in range(0, int(self.len[1] / self.dw[2])):
                    for k in range(0, int(self.len[2] / self.dw[3])):
                        self.func_dim3[i, j, k] = f1[i][j][k]

    @ti.kernel
    def height_solver_dim1(self):
        # 初始化边界条件
        a1 = self.a[0]
        b1 = self.b[0]
        v = self.v
        t = self.t
        dt = self.dw[0]
        dx = self.dw[1]
        len1 = self.len[0]
        self.phi_xt[0, 0] = a1
        self.phi_xt[int(len1 / dx) - 1, 0] = b1
        for i in ti.ndrange((1, int(len1 / dx) - 1)):
            self.phi_xt[i, 0] = self.func_dim1[i]  # 初始位移条件
        for n in ti.ndrange((0, 513)):  # 假设 self.t 是总时间步数
            self.phi_xt[0, n + 1] = a1
            self.phi_xt[int(len1 / dx) - 1, n + 1] = b1
            for i in ti.ndrange((1, int(len1 / dx) - 1)):
                self.phi_xt[i, n + 1] = self.phi_xt[i, n] + v * dt / (dx ** 2) * (
                        self.phi_xt[i + 1, n] - 2 * self.phi_xt[i, n] + self.phi_xt[i - 1, n])
        for n in ti.ndrange((513, int(t / dt) - 1)):  # 假设 self.t 是总时间步数
            self.phi_xt[0, n + 1] = a1
            self.phi_xt[int(len1 / dx) - 1, n + 1] = b1
            for i in ti.ndrange((1, int(len1 / dx) - 1)):
                self.phi_xt[i, n + 1] = self.phi_xt[i, n] + v * dt / (dx ** 2) * (
                        self.phi_xt[i + 1, n] - 2 * self.phi_xt[i, n] + self.phi_xt[i - 1, n])

    @ti.kernel
    def height_solver_dim2(self):
        a1, a2 = self.a[0], self.a[1]  # 沿x和y的边界条件
        b1, b2 = self.b[0], self.b[1]  # 沿x和y的边界条件
        v = self.v
        dt = self.dw[0]
        dx = self.dw[1]
        dy = self.dw[2]
        t = self.t
        len1, len2 = self.len[0], self.len[1]
        for i, j in ti.ndrange((0, int(len1 / dx) - 1), (0, int(len2 / dy) - 1)):
            self.phi_xyt[i, j, 0] = self.func_dim2[i, j]
        # 边界条件初始化，对于二维情况，需要设置四边的边界
        for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
            self.phi_xyt[0, j, 0] = a1
            self.phi_xyt[int(len1 / dx) - 1, j, 0] = b1
        for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
            self.phi_xyt[i, 0, 0] = a2
            self.phi_xyt[i, int(len2 / dy) - 1, 0] = b2
            # 迭代更新
        for n in ti.ndrange((0, 511)):  # 假设 self.t 是总时间步数int(t / dt) - 1)
            for i, j in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1)):
                self.phi_xyt[i, j, n + 1] = self.phi_xyt[i, j, n] + v * dt / (dx ** 2) * (
                        self.phi_xyt[i + 1, j, n] +
                        self.phi_xyt[i - 1, j, n] +
                        self.phi_xyt[i, j + 1, n] +
                        self.phi_xyt[i, j - 1, n] -
                        4 * self.phi_xyt[i, j, n]
                )
            # 更新边界条件以保持不变
            for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
                self.phi_xyt[0, j, n] = a1
                self.phi_xyt[int(len1 / dx) - 1, j, n] = b1
            for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
                self.phi_xyt[i, 0, n] = a2
                self.phi_xyt[i, int(len2 / dy) - 1, n] = b2
        for n in ti.ndrange((511,1024)):  # 假设 self.t 是总时间步数
            for i, j in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1)):
                self.phi_xyt[i, j, n + 1] = self.phi_xyt[i, j, n] + v * dt / (dx ** 2) * (
                        self.phi_xyt[i + 1, j, n] +
                        self.phi_xyt[i - 1, j, n] +
                        self.phi_xyt[i, j + 1, n] +
                        self.phi_xyt[i, j - 1, n] -
                        4 * self.phi_xyt[i, j, n]
                )
            # 更新边界条件以保持不变
            for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
                self.phi_xyt[0, j, n] = a1
                self.phi_xyt[int(len1 / dx) - 1, j, n] = b1
            for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
                self.phi_xyt[i, 0, n] = a2
                self.phi_xyt[i, int(len2 / dy) - 1, n] = b2
        for n in ti.ndrange((1024,int(t / dt) - 1 )):  # 假设 self.t 是总时间步数
            for i, j in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1)):
                self.phi_xyt[i, j, n + 1] = self.phi_xyt[i, j, n] + v * dt / (dx ** 2) * (
                        self.phi_xyt[i + 1, j, n] +
                        self.phi_xyt[i - 1, j, n] +
                        self.phi_xyt[i, j + 1, n] +
                        self.phi_xyt[i, j - 1, n] -
                        4 * self.phi_xyt[i, j, n]
                )
            # 更新边界条件以保持不变
            for j in ti.ndrange((0, int(len2 / dy))):  # x方向的边界
                self.phi_xyt[0, j, n] = a1
                self.phi_xyt[int(len1 / dx) - 1, j, n] = b1
            for i in ti.ndrange((0, int(len1 / dx))):  # y方向的边界
                self.phi_xyt[i, 0, n] = a2
                self.phi_xyt[i, int(len2 / dy) - 1, n] = b2


    @ti.kernel
    def height_solver_dim3(self):
        # 定义边界条件
        a1, a2, a3 = self.a[0], self.a[1], self.a[2]
        b1, b2, b3 = self.b[0], self.b[1], self.b[2]
        v = self.v
        dt = self.dw[0]
        dx = self.dw[1]
        dy = self.dw[2]
        dz = self.dw[3]
        len1, len2, len3 = self.len[0], self.len[1], self.len[2]
        t = self.t
        # 应用初始条件
        for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
            self.phi_xyzt[i, j, k, 0] = self.func_dim3[i, j, k]
            # 在第一次迭代之前固定边界条件
        for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
            if i == 0 or i == int(len1 / dx) - 1:
                self.phi_xyzt[i, j, k, 0] = a1 if i == 0 else b1
            if j == 0 or j == int(len2 / dy) - 1:
                self.phi_xyzt[i, j, k, 0] = a2 if j == 0 else b2
            if k == 0 or k == int(len3 / dz) - 1:
                self.phi_xyzt[i, j, k, 0] = a3 if k == 0 else b3
        for n in ti.ndrange((0, 511)):  # 假设 self.t 是总时间步数
            for i, j, k in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1), (1, int(len3 / dz) - 1)):
                self.phi_xyzt[i, j, k, n + 1] = self.phi_xyzt[i, j, k, n] + v * dt / (dx ** 2) * (
                        self.phi_xyzt[i + 1, j, k, n] +
                        self.phi_xyzt[i - 1, j, k, n] +
                        self.phi_xyzt[i, j + 1, k, n] +
                        self.phi_xyzt[i, j - 1, k, n] +
                        self.phi_xyzt[i, j, k + 1, n] +
                        self.phi_xyzt[i, j, k - 1, n] -
                        6 * self.phi_xyzt[i, j, k, n]
                )
            # 在每次迭代之后重新固定边界条件
            for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
                if i == 0 or i == int(len1 / dx) - 1:
                    self.phi_xyzt[i, j, k, t] = a1 if i == 0 else b1
                if j == 0 or j == int(len2 / dy) - 1:
                    self.phi_xyzt[i, j, k, t] = a2 if j == 0 else b2
                if k == 0 or k == int(len3 / dz) - 1:
                    self.phi_xyzt[i, j, k, t] = a3 if k == 0 else b3
        for n in ti.ndrange((511, int(self.t/dt)-1)):  # 假设 self.t 是总时间步数
            for i, j, k in ti.ndrange((1, int(len1 / dx) - 1), (1, int(len2 / dy) - 1), (1, int(len3 / dz) - 1)):
                self.phi_xyzt[i, j, k, n + 1] = self.phi_xyzt[i, j, k, n] + v * dt / (dx ** 2) * (
                        self.phi_xyzt[i + 1, j, k, n] +
                        self.phi_xyzt[i - 1, j, k, n] +
                        self.phi_xyzt[i, j + 1, k, n] +
                        self.phi_xyzt[i, j - 1, k, n] +
                        self.phi_xyzt[i, j, k + 1, n] +
                        self.phi_xyzt[i, j, k - 1, n] -
                        6 * self.phi_xyzt[i, j, k, n]
                )
            # 在每次迭代之后重新固定边界条件
            for i, j, k in ti.ndrange((0, int(len1 / dx)), (0, int(len2 / dy)), (0, int(len3 / dz))):
                if i == 0 or i == int(len1 / dx) - 1:
                    self.phi_xyzt[i, j, k, t] = a1 if i == 0 else b1
                if j == 0 or j == int(len2 / dy) - 1:
                    self.phi_xyzt[i, j, k, t] = a2 if j == 0 else b2
                if k == 0 or k == int(len3 / dz) - 1:
                    self.phi_xyzt[i, j, k, t] = a3 if k == 0 else b3
    def save_data_dim1(self):
        data = self.phi_xt.to_numpy()
        np.save('height_dim1_data.npy',data)

    def load_data_dim1(self):
        data = np.load('height_dim1_data.npy')
        temp_phi_xt = ti.field(dtype=ti.f32, shape=data.shape)
        temp_phi_xt.from_numpy(data)
        self.phi_xt = temp_phi_xt
    def save_data_dim2(self):
        data = self.phi_xyt.to_numpy()
        np.save('height_dim2_data.npy',data)

    def load_data_dim2(self):
        data = np.load('height_dim2_data.npy')
        temp_phi_xyt = ti.field(dtype=ti.f32, shape=data.shape)
        temp_phi_xyt.from_numpy(data)
        self.phi_xyt = temp_phi_xyt
    def save_data_dim3(self):
        data = self.phi_xyzt.to_numpy()
        np.save('height_dim3_data.npy',data)

    def load_data_dim3(self):
        data = np.load('height_dim3_data.npy')
        temp_phi_xyzt = ti.field(dtype=ti.f32, shape=data.shape)
        temp_phi_xyzt.from_numpy(data)
        self.phi_xyzt = temp_phi_xyzt
    def print_field_dim1(self):
        for i, j in ti.ndrange((0, int(self.len[0] / self.dw[1])), (0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j}] = {self.phi_xt[i, j]}")  # 打印field的每个元素

    def print_field_dim2(self):
        for i, j, k in ti.ndrange((0, int(self.len[0] / self.dw[1])), (0, int(self.len[1] / self.dw[2])),
                                  (0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j},{k}] = {self.phi_xyt[i, j, k]}")  # 打印field的每个元素

    def print_field_dim3(self):
        for i, j, k, t in ti.ndrange((0, int(self.len[0] / self.dw[1])), (0, int(self.len[1] / self.dw[2])),
                                     (0, int(self.len[2] / self.dw[3])), (0, int(self.t / self.dw[0]))):
            print(f"x[{i}, {j},{k},{t}] = {self.phi_xyzt[i, j, k, t]}")  # 打印field的每个元素

    def draw_dim1(self):
        dx = self.dw[1]
        dt = self.dw[0]
        len1 = self.len[0]
        t = self.t
        len_points = int(len1 / dx)
        time_points = int(t / dt)
        data_ti = self.phi_xt
        gui = ti.GUI("弦振动动画", res=(800, 600))
        for i, j in ti.ndrange((0, int(self.len[0] / self.dw[1])), (0, int(self.t / self.dw[0]))):
            data_ti[i,j] = data_ti[i,j]+15
            print(int(self.len[0] / self.dw[1])-i)
        max_value = max(data_ti.to_numpy().flatten())  # 提前计算最大值以提高效率

        for t in range(time_points):
<<<<<<< HEAD
            if t == 100:
                break
=======
>>>>>>> b847bb8b2be910bde0f6aea8e589dce5a393d6b1
            pos = []
            for i in range(len_points):
                # 假设 data_ti 中存储的y值需要被适当缩放和位移以适配GUI的显示范围
                x = i / len_points  # 将 x 坐标标准化到 [0, 1]
                y = data_ti[i, t] / max_value  # 将 y 值归一化
                pos.append([x, y])
            pos_np = np.array(pos)  # 将 pos 列表转换为 numpy 数组
            gui.circles(pos_np, radius=2, color=0x068587)
            gui.show(str(t) + '.png')

    def exe_dim1_solver(self):
        self.init_func()
        self.height_solver_dim1()
        self.save_data_dim1()

    def exe_dim2_solver(self):
        self.init_func()
        self.height_solver_dim2()
        self.save_data_dim2()

    def exe_dim3_solver(self):
        self.init_func()
        self.height_solver_dim3()
        self.save_data_dim3()

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    # a = height_equ("3\sinx",[10,5,5],1,0.1,[0.01,0.1,0.1,0.1],[0,0,0],[0,0,0],11)
    # a.exe_dim1_solver()
    a = height_equ("\sin(x+y)", [10, 5, 5], 2, 0.1, [0.01, 0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0], 20)
    a.exe_dim2_solver()
    # a = height_equ("\sin(x+y+z)",[10,5,5],3,0.1,[0.01,0.1,0.1,0.1],[0,0,0],[0,0,0],11)
    # a.exe_dim3_solver()
