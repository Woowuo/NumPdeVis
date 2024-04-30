
import latex2sympy2
import numpy as np
import sympy as sp
from re import findall, split, MULTILINE

from matplotlib import pyplot as plt
from sympy import solve as solving, latex, sympify, Matrix, re
from latex2sympy2 import latex2sympy, latex2latex
from mpl_toolkits.mplot3d import Axes3D


class calculator_normal():
    def __init__(self, latex_string = None,model = None):
        self.func_strings = []
        self.func_strings.append(latex_string)
        # self.model = model

    def solve_non_equations(self,latex_text, formatter='latex'):
        regex = r"\\begin{cases}([\s\S]*)\\end{cases}"
        matches = findall(regex, latex_text, MULTILINE)
        equations = []
        if matches:
            matches = split(r"\\\\(?:\[?.*?\])?", matches[0])
            for match in matches:
                ins = latex2sympy(match)
                if type(ins) == list:
                    equations.extend(ins)
                else:
                    equations.append(ins)
            solved = solving(equations)
        else:
            return False
        if formatter == 'latex':
            return latex(solved)
        else:
            return solved
    def add_string(self,string):
        self.func_strings.append(string)

    def nonlinear_equation_solver(self,string):
        x = sp.symbols('x')
        equation = latex2sympy2.latex2sympy(string)
        derivative = sp.diff(equation, x)

        f = sp.lambdify(x, equation, "numpy")
        f_prime = sp.lambdify(x, derivative, "numpy")

        roots = []
        num_guesses = 10
        tolerance = 1e-10
        max_iterations = 1000
        unique_tolerance = 1e-5

        for _ in range(num_guesses):
            x_n = np.random.uniform(-100, 100)
            iteration = 0

            while True:
                f_prime_val = f_prime(x_n)
                if abs(f_prime_val) < tolerance:  # 避免除以零
                    break
                x_n1 = x_n - f(x_n) / f_prime_val
                if abs(x_n1 - x_n) < tolerance or iteration >= max_iterations:
                    # 检查新解是否与已找到的解足够不同
                    if not any(abs(x_n1 - root) < unique_tolerance for root in roots):
                        roots.append(x_n1)
                    break
                x_n = x_n1
                iteration += 1

        return latex2sympy2.latex(roots)

    def cal_func_dri(self,str_f_expr):
        """
        计算一元函数的导数并返回一个可调用的Python函数

        参数:
        - f_expr: sympy表达式，表示一元函数
        - x: sympy符号，表示函数的变量

        返回:
        - 导数的Python函数
        """
        # 计算导数
        f_expr = latex2sympy(str_f_expr)
        x = sp.symbols('x')
        f_prime_expr = sp.diff(f_expr, x)

        # 将符号表达式转换为可调用的Python函数
        f_prime_func = sp.lambdify(x, f_prime_expr, 'numpy')

        return latex2sympy2.latex(f_prime_func)
    def find_extreme(self,str_f_expr):
        """
        计算一元函数的极值点

        参数:
        - f_expr: sympy表达式，表示一元函数
        - x: sympy符号，表示函数的变量

        返回:
        - 极值点列表
        """
        # 计算一阶导数
        f_expr = latex2sympy(str_f_expr)
        x = sp.symbols('x')
        f_prime = sp.diff(f_expr, x)

        # 解一阶导数等于0的方程，找到可能的极值点
        critical_points = sp.solve(f_prime, x)

        # （可选）计算二阶导数并检查每个临界点
        f_double_prime = sp.diff(f_prime, x)
        for point in critical_points:
            # 二阶导数检验
            if f_double_prime.subs(x, point) > 0:
                print(f"{point} 是极小值点")
            elif f_double_prime.subs(x, point) < 0:
                print(f"{point} 是极大值点")
            else:
                print(f"{point} 是鞍点")

        return critical_points
    def solve_integral(self,string):
        try:
            # 使用latex2sympy2将LaTeX字符串转换为sympy表达式
            expression = latex2sympy2.latex2sympy(string)
            # 定义变量x
            x = sp.symbols('x')

            # 检查表达式是否为定积分（有边界）
            if isinstance(expression, sp.Integral):
                if len(expression.limits) == 1 and len(expression.limits[0]) == 3:
                    # 提取积分变量和积分区间（定积分）
                    _, a, b = expression.limits[0]
                    solution = sp.integrate(expression.function, (x, a, b))
                else:
                    # 处理不定积分
                    solution = expression.doit()
                    solution = latex2sympy2.latex(solution)
            else:
                # 若表达式不是积分，则直接返回
                return "表达式不是一个积分"
            print(solution)
            return solution
        except Exception as e:
            # 处理可能出现的任何异常，并返回错误信息
            return f"发生错误: {str(e)}"


    def normal_solve(self,string):
        func = latex2sympy2.latex2sympy(string)
        # func.evalf
        return str(func.evalf())

    # \=\\
    def calculate_matrix(self,latex_str):
        # 将 LaTeX 字符串转换为 SymPy 表达式
        expr = latex2sympy(latex_str)

        # 假设转换后的表达式是一个矩阵求和操作
        # 计算这个表达式
        result = expr.doit()
        result = latex2sympy2.latex2latex(result)
        return result

    from latex2sympy2 import latex2sympy
    from sympy import Eq, solve, symbols

    def solve_linear_system_from_latex(self,str_A,str_b):
        # 解析 LaTeX 字符串
        A = latex2sympy(str_A)
        b = latex2sympy(str_b)
        n = A.shape[1]  # A的列数
        X_symbols = sp.symbols(f'x1:{n + 1}')  # 创建n个符号变量

        # 使用linsolve求解
        solution = sp.linsolve((A, b), X_symbols)
        return latex2sympy2.latex(solution)

    def inverse_matrix(self,A_str):
        """
        返回给定SymPy矩阵对象A的逆矩阵。

        参数:
        - A: 一个SymPy矩阵对象。

        返回:
        - A的逆矩阵，如果存在的话；否则，如果A不可逆，抛出异常。
        """
        A = latex2sympy(A_str)
        # 检查A是否为方阵
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须是方阵才能求逆。")

        # 检查A是否可逆（即行列式不为0）
        if A.det() == 0:
            raise ValueError("矩阵不可逆，因为其行列式为0。")

        # 返回逆矩阵
        return latex2sympy2.latex(A.inv())
    def plot_from_latex_multiple(self):
        plt.style.use('ggplot')  # 选择一个炫酷的图形样式
        plt.figure(figsize=(10, 6))  # 调整图形的大小

        # 定义变量 x
        x = sp.symbols('x')
        x_vals = np.linspace(-10, 10, 400)  # 定义 x 轴上的点

        for string in self.func_strings:
            expr = latex2sympy2.latex2sympy(string)
            f = sp.lambdify(x, expr, 'numpy')
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label='$' + sp.latex(expr) + '$', linestyle='-', linewidth=2)

        plt.fill_between(x_vals, 0, y_vals, color="skyblue", alpha=0.4)  # 添加填充效果
        plt.legend(loc='best')
        plt.grid(True)  # 显示网格
        plt.title('函数图像')  # 添加标题
        plt.xlabel('$x$')  # 添加 x 轴标签
        plt.ylabel('$f(x)$')  # 添加 y 轴标签
        plt.show()

    def draw_binary_func(self,str_f_expr):
        """
        绘制二元函数的等高线图。

        参数:
        - f_expr: sympy表达式，表示二元函数。
        - x_range: (min, max)形式的元组，定义x的范围。
        - y_range: (min, max)形式的元组，定义y的范围。
        - title: 图形的标题。
        """
        f_expr = latex2sympy(str_f_expr)
        # 定义符号变量
        x, y = sp.symbols('x y')

        # 将sympy表达式转换为可用于numpy的函数
        f_lambdified = sp.lambdify((x, y), f_expr, 'numpy')

        # 创建一个网格
        x_vals = np.linspace(-100, 100, 1000)
        y_vals = np.linspace(-100, 100, 1000)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 在网格上计算函数值
        Z = f_lambdified(X, Y)

        # 绘制图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面图
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # 添加色彩条

        # 设置标题和轴标签
        ax.set_title("diagram")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')

        # 显示图形
        plt.show()

    def solve_diff_equ(self,str_func):
        x = sp.symbols('x')
        f = sp.Function('f')
        func = latex2sympy(str_func)
        solve = sp.dsolve(func,f(x))
        return latex2sympy2.latex(solve)



    def exe_test(self):
        self.solve_integral(self.func_strings[0])

if __name__ == '__main__':
    equation_solver = calculator_normal("\int\cosxdx", 0)

    equation_solver.add_string(r"x^2+y^2")

    equation_solver.exe_test()



import latex2sympy2
import numpy as np
import sympy as sp
from re import findall, split, MULTILINE

from matplotlib import pyplot as plt
from sympy import solve as solving, latex, sympify, Matrix, re
from latex2sympy2 import latex2sympy, latex2latex
from mpl_toolkits.mplot3d import Axes3D


class calculator_normal():
    def __init__(self, latex_string = None,model = None):
        self.func_strings = []
        self.func_strings.append(latex_string)
        self.model = model

    def solve_non_equations(self,latex_text, formatter='latex'):
        regex = r"\\begin{cases}([\s\S]*)\\end{cases}"
        matches = findall(regex, latex_text, MULTILINE)
        equations = []
        if matches:
            matches = split(r"\\\\(?:\[?.*?\])?", matches[0])
            for match in matches:
                ins = latex2sympy(match)
                if type(ins) == list:
                    equations.extend(ins)
                else:
                    equations.append(ins)
            solved = solving(equations)
        else:
            return False
        if formatter == 'latex':
            return latex(solved)
        else:
            return solved
    def add_string(self,string):
        self.func_strings.append(string)

    def nonlinear_equation_solver(self,string):
        x = sp.symbols('x')
        equation = latex2sympy2.latex2sympy(string)
        derivative = sp.diff(equation, x)

        f = sp.lambdify(x, equation, "numpy")
        f_prime = sp.lambdify(x, derivative, "numpy")

        roots = []
        num_guesses = 10
        tolerance = 1e-10
        max_iterations = 1000
        unique_tolerance = 1e-5

        for _ in range(num_guesses):
            x_n = np.random.uniform(-100, 100)
            iteration = 0

            while True:
                f_prime_val = f_prime(x_n)
                if abs(f_prime_val) < tolerance:  # 避免除以零
                    break
                x_n1 = x_n - f(x_n) / f_prime_val
                if abs(x_n1 - x_n) < tolerance or iteration >= max_iterations:
                    # 检查新解是否与已找到的解足够不同
                    if not any(abs(x_n1 - root) < unique_tolerance for root in roots):
                        roots.append(x_n1)
                    break
                x_n = x_n1
                iteration += 1

        return roots

    def cal_func_dri(self,str_f_expr):
        """
        计算一元函数的导数并返回一个可调用的Python函数

        参数:
        - f_expr: sympy表达式，表示一元函数
        - x: sympy符号，表示函数的变量

        返回:
        - 导数的Python函数
        """
        # 计算导数
        f_expr = latex2sympy(str_f_expr)
        x = sp.symbols('x')
        f_prime_expr = sp.diff(f_expr, x)

        # 将符号表达式转换为可调用的Python函数
        f_prime_func = sp.lambdify(x, f_prime_expr, 'numpy')

        return f_prime_func
    def find_extreme(self,str_f_expr):
        """
        计算一元函数的极值点

        参数:
        - f_expr: sympy表达式，表示一元函数
        - x: sympy符号，表示函数的变量

        返回:
        - 极值点列表
        """
        # 计算一阶导数
        f_expr = latex2sympy(str_f_expr)
        x = sp.symbols('x')
        f_prime = sp.diff(f_expr, x)

        # 解一阶导数等于0的方程，找到可能的极值点
        critical_points = sp.solve(f_prime, x)

        # （可选）计算二阶导数并检查每个临界点
        f_double_prime = sp.diff(f_prime, x)
        for point in critical_points:
            # 二阶导数检验
            if f_double_prime.subs(x, point) > 0:
                print(f"{point} 是极小值点")
            elif f_double_prime.subs(x, point) < 0:
                print(f"{point} 是极大值点")
            else:
                print(f"{point} 是鞍点")

        return critical_points
    def solve_integral(self,string):
        try:
            # 使用latex2sympy2将LaTeX字符串转换为sympy表达式
            expression = latex2sympy2.latex2sympy(string)
            # 定义变量x
            x = sp.symbols('x')

            # 检查表达式是否为定积分（有边界）
            if isinstance(expression, sp.Integral):
                if len(expression.limits) == 1 and len(expression.limits[0]) == 3:
                    # 提取积分变量和积分区间（定积分）
                    _, a, b = expression.limits[0]
                    solution = sp.integrate(expression.function, (x, a, b))
                else:
                    # 处理不定积分
                    solution = expression.doit()
                    solution = latex2sympy2.latex(solution)
            else:
                # 若表达式不是积分，则直接返回
                return "表达式不是一个积分"

            return solution
        except Exception as e:
            # 处理可能出现的任何异常，并返回错误信息
            return f"发生错误: {str(e)}"


    def normal_solve(self,string):
        func = latex2sympy2.latex2sympy(string)
        # func.evalf
        return func.evalf()

    # \=\\
    def calculate_matrix(self,latex_str):
        # 将 LaTeX 字符串转换为 SymPy 表达式
        expr = latex2sympy(latex_str)

        # 假设转换后的表达式是一个矩阵求和操作
        # 计算这个表达式
        result = expr.doit()

        return result

    from latex2sympy2 import latex2sympy
    from sympy import Eq, solve, symbols

    def solve_linear_system_from_latex(self,str_A,str_b):
        # 解析 LaTeX 字符串
        A = latex2sympy(str_A)
        b = latex2sympy(str_b)
        n = A.shape[1]  # A的列数
        X_symbols = sp.symbols(f'x1:{n + 1}')  # 创建n个符号变量

        # 使用linsolve求解
        solution = sp.linsolve((A, b), X_symbols)
        return solution

    def inverse_matrix(self,A_str):
        """
        返回给定SymPy矩阵对象A的逆矩阵。

        参数:
        - A: 一个SymPy矩阵对象。

        返回:
        - A的逆矩阵，如果存在的话；否则，如果A不可逆，抛出异常。
        """
        A = latex2sympy(A_str)
        # 检查A是否为方阵
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须是方阵才能求逆。")

        # 检查A是否可逆（即行列式不为0）
        if A.det() == 0:
            raise ValueError("矩阵不可逆，因为其行列式为0。")

        # 返回逆矩阵
        return A.inv()
    def plot_from_latex_multiple(self):
        plt.style.use('ggplot')  # 选择一个炫酷的图形样式
        plt.figure(figsize=(10, 6))  # 调整图形的大小

        # 定义变量 x
        x = sp.symbols('x')
        x_vals = np.linspace(-10, 10, 400)  # 定义 x 轴上的点

        for string in self.func_strings:
            expr = latex2sympy2.latex2sympy(string)
            f = sp.lambdify(x, expr, 'numpy')
            y_vals = f(x_vals)
            plt.plot(x_vals, y_vals, label='$' + sp.latex(expr) + '$', linestyle='-', linewidth=2)

        plt.fill_between(x_vals, 0, y_vals, color="skyblue", alpha=0.4)  # 添加填充效果
        plt.legend(loc='best')
        plt.grid(True)  # 显示网格
        plt.title('函数图像')  # 添加标题
        plt.xlabel('$x$')  # 添加 x 轴标签
        plt.ylabel('$f(x)$')  # 添加 y 轴标签
        plt.show()

    def draw_binary_func(self,str_f_expr):
        """
        绘制二元函数的等高线图。

        参数:
        - f_expr: sympy表达式，表示二元函数。
        - x_range: (min, max)形式的元组，定义x的范围。
        - y_range: (min, max)形式的元组，定义y的范围。
        - title: 图形的标题。
        """
        f_expr = latex2sympy(str_f_expr)
        # 定义符号变量
        x, y = sp.symbols('x y')

        # 将sympy表达式转换为可用于numpy的函数
        f_lambdified = sp.lambdify((x, y), f_expr, 'numpy')

        # 创建一个网格
        x_vals = np.linspace(-100, 100, 1000)
        y_vals = np.linspace(-100, 100, 1000)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 在网格上计算函数值
        Z = f_lambdified(X, Y)

        # 绘制图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面图
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # 添加色彩条

        # 设置标题和轴标签
        ax.set_title("diagram")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')

        # 显示图形
        plt.show()

    def solve_diff_equ(self,str_func):
        x = sp.symbols('x')
        f = sp.Function('f')
        func = latex2sympy(str_func)
        solve = sp.dsolve(func,f(x))
        return solve



    def exe_test(self):
        print(self.solve_diff_equ(self.func_strings[1]))

if __name__ == '__main__':
    equation_solver = calculator_normal("\\begin{cases} 5x+4 \\geqslant 2(x-1) \\\\\\\\[1.5ex] \\dfrac{2x+5}{3}-\\dfrac{3x-2}{2}>1 \\end{cases}"
    , 0)
    equation_solver.add_string(r"\frac{d}{dx}f(x)-f(x)")
    equation_solver.exe_test()



