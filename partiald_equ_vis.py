import imageio
import taichi as ti
import numpy as np
import wave_equ_solver as wa
import height_equ_solver as he
import Record as re
import subprocess


ti.init(arch=ti.gpu)
model = 12
if model == 11:
    a = wa.wave_equ("\sinx", "\cosx", [10, 5, 5], 1, 0.1, [0.05, 0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0], 100)
    a.load_data_dim1()
    a.draw_dim1()
    subprocess.run(['python', 'Record.py'], check=True)

if model == 12:
    a = wa.wave_equ("\sinx", "\sinx", [10, 5, 5], 2, 0.1, [0.1, 0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0], 100)
    a.load_data_dim2()
    Nx = int(a.len[0] / a.dw[1])  # 网格大小
    Ny = int(a.len[1] / a.dw[2])
    dx = a.dw[1]  # 网格间距
    dy = a.dw[2]
    dt = a.dw[0]  # 时间步长

    # Taichi变量
    scalar = ti.field
    height = scalar(dtype=ti.f32, shape=(Nx, Ny))
    velocity = scalar(dtype=ti.f32, shape=(Nx, Ny))
    colors = ti.Vector.field(3, dtype=ti.f32, shape=(Nx, Ny))

    # 粒子位置和颜色（一维表示）
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=(Nx * Ny))
    particles_color = ti.Vector.field(3, dtype=ti.f32, shape=(Nx * Ny))


    # 初始化波动
    @ti.kernel
    def initialize_wave():
        for i, j in ti.ndrange(Nx, Ny):
            height[i, j] = a.phi_xyt[i, j, 0]


    # 更新波动方程
    @ti.kernel
    def update_wave(t: ti.f32):
        for i, j in ti.ndrange((1, Nx - 1), (1, Ny - 1)):  # 分别为i和j维度设置范围
            height[i, j] = a.phi_xyt[i, j, int(t)]


    # 将波动高度数据转换为颜色
    @ti.kernel
    def fill_colors():
        for i, j in height:
            if (i // 32 + j // 32) % 2 == 0:  # 调整间隔大小以适应粒子的密度
                colors[i, j] = ti.Vector([0.22, 0.72, 0.52])
            else:
                colors[i, j] = ti.Vector([1, 0.334, 0.52])


    # 二维转一维
    @ti.kernel
    def flatten_fields():
        for i, j in ti.ndrange(Nx, Ny):
            idx = i * Nx + j
            particles_pos[idx] = ti.Vector([i * dx - 0.5, height[i, j], j * dy - 0.5])
            particles_color[idx] = colors[i, j]


    window = ti.ui.Window('2D Wave Equation', (800, 800))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    # 设置摄像机
    camera = ti.ui.make_camera()
    camera.position(-15, 15, 0)
    camera.lookat(2, 0, 2)
    camera.fov(20)
    initialize_wave()
    t = 0

    while window.running:
        if t == 100:
            break

        t = t + 1
        update_wave(t)

        fill_colors()
        flatten_fields()  # 更新粒子位置和颜色

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)

        scene.set_camera(camera)

        # 设置点光源和环境光源
        scene.point_light(pos=(1, 2, 2), color=(0.8, 0.8, 1))  # 调整光源颜色为偏蓝色调
        scene.ambient_light((0.2, 0.2, 0.6))  # 调整环境光为偏蓝色调
        # 使用颜色渲染场景
        scene.particles(particles_pos, radius=0.02, per_vertex_color=particles_color)  # 可以调整粒子半径
        print(particles_pos)
        # 渲染场景
        canvas.scene(scene)
        window.save_image(str(t)+'.png')
        window.show()
    subprocess.run(['python', 'Record.py'], check=True)

if model == 21:
    a = he.height_equ("3\sinx",[10,5,5],1,0.1,[0.01,0.1,0.1,0.1],[0,0,0],[0,0,0],11)
    a.load_data_dim1()
    a.draw_dim1()
if model == 22:
    a = he.height_equ("\sin(x+y)", [10, 5, 5], 2, 0.1, [0.01, 0.1, 0.1, 0.1], [0, 0, 0], [0, 0, 0], 11)
    a.load_data_dim2()
    Nx = int(a.len[0] / a.dw[1])  # 网格大小
    Ny = int(a.len[1] / a.dw[2])
    dx = a.dw[1]  # 网格间距
    dy = a.dw[2]
    dt = a.dw[0]  # 时间步长
    result_dir = "./results"
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    # Taichi变量
    scalar = ti.field
    height = scalar(dtype=ti.f32, shape=(Nx, Ny))
    velocity = scalar(dtype=ti.f32, shape=(Nx, Ny))
    colors = ti.Vector.field(3, dtype=ti.f32, shape=(Nx, Ny))

    # 粒子位置和颜色（一维表示）
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=(Nx * Ny))
    particles_color = ti.Vector.field(3, dtype=ti.f32, shape=(Nx * Ny))


    # 初始化波动
    @ti.kernel
    def initialize_wave():
        for i, j in ti.ndrange(Nx, Ny):
            height[i, j] = a.phi_xyt[i, j, 0]


    # 更新波动方程
    @ti.kernel
    def update_wave(t: ti.f32):
        for i, j in ti.ndrange((1, Nx - 1), (1, Ny - 1)):  # 分别为i和j维度设置范围
            height[i, j] = a.phi_xyt[i, j, int(t)]


    # 将波动高度数据转换为颜色
    @ti.kernel
    def fill_colors():
        for i, j in height:
            if (i // 32 + j // 32) % 2 == 0:  # 调整间隔大小以适应粒子的密度
                colors[i, j] = ti.Vector([0.22, 0.72, 0.52])
            else:
                colors[i, j] = ti.Vector([1, 0.334, 0.52])


    # 二维转一维
    @ti.kernel
    def flatten_fields():
        for i, j in ti.ndrange(Nx, Ny):
            idx = i * Nx + j
            particles_pos[idx] = ti.Vector([i * dx - 0.5, height[i, j], j * dy - 0.5])
            particles_color[idx] = colors[i, j]


    window = ti.ui.Window('2D  Equation', (800, 800),show_window=True)
    window = ti.ui.Window('2D  Equation', (800, 800),show_window=False)

    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()

    # 设置摄像机
    camera = ti.ui.make_camera()
    camera.position(-15, 15, 0)
    camera.lookat(2, 0, 2)
    camera.fov(20)

    initialize_wave()
    t = 0
    images = []  # 存储图像帧

    while window.running:
        if t == 100:
            break
        t = t + 1
        update_wave(t)

        fill_colors()
        flatten_fields()  # 更新粒子位置和颜色

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # 设置点光源和环境光源
        scene.point_light(pos=(1, 2, 2), color=(0.8, 0.8, 1))  # 调整光源颜色为偏蓝色调
        scene.ambient_light((0.2, 0.2, 0.6))  # 调整环境光为偏蓝色调
        # 使用颜色渲染场景
        scene.particles(particles_pos, radius=0.02, per_vertex_color=particles_color)  # 可以调整粒子半径
        print(particles_pos)
        # 渲染场景
        canvas.scene(scene)
        print("1")
        window.save_image(str(t)+'.png')
        window.show()