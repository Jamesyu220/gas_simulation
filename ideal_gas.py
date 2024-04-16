import taichi as ti
import numpy as np
import math
import tkinter as tk
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

from src.particles_motion import particle_motion, particle_collision, border_collisions, emit_from_drain
from src.particle_source import add_diffusion_particles
from src.temperature import get_v_abs, cal_temperature

box_size = 0.8
drain_size = 0.05
xmin = -box_size
xmax = box_size
ymin = -box_size
ymax = box_size
zmin = -box_size
zmax = box_size

line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(8,))
line_vertices[0] = [xmin, ymax, zmin]
line_vertices[1] = [xmax, ymax, zmin]
line_vertices[2] = [xmax, ymax, zmax]
line_vertices[3] = [xmin, ymax, zmax]
line_vertices[4] = [xmin, ymin, zmin]
line_vertices[5] = [xmax, ymin, zmin]
line_vertices[6] = [xmax, ymin, zmax]
line_vertices[7] = [xmin, ymin, zmax]

line_indices = ti.field(dtype=ti.i32, shape=(24,))
line_indices.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4]))

heater_length = 0.8  # Length to match the width of the box
heater_width = 0.8   # Width to match the depth of the box
heater_height = 0.05      # Height of the heater box as specified
heater_gap = 0.1

# Heater size setting and center it right below the box.
# Calculate the starting positions (xmin and zmin for the heater) to center it
heater_xmin = (xmin + xmax - heater_length) / 2
heater_zmin = (zmin + zmax - heater_width) / 2
heater_ymin = ymin - heater_gap - heater_height  # Y position of the heater's bottom
heater_xmax = heater_xmin + heater_length
heater_zmax = heater_zmin + heater_width

# Heater vertices (centered below the simulation box)
heater_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(8,))
heater_vertices[0] = [heater_xmin, heater_ymin, heater_zmin]
heater_vertices[1] = [heater_xmin + heater_length, heater_ymin, heater_zmin]
heater_vertices[2] = [heater_xmin + heater_length, heater_ymin, heater_zmin + heater_width]
heater_vertices[3] = [heater_xmin, heater_ymin, heater_zmin + heater_width]
heater_vertices[4] = [heater_xmin, heater_ymin + heater_height, heater_zmin]
heater_vertices[5] = [heater_xmin + heater_length, heater_ymin + heater_height, heater_zmin]
heater_vertices[6] = [heater_xmin + heater_length, heater_ymin + heater_height, heater_zmin + heater_width]
heater_vertices[7] = [heater_xmin, heater_ymin + heater_height, heater_zmin + heater_width]

# Heater indices for line rendering (to draw a cube)
heater_indices = ti.field(dtype=ti.i32, shape=(24,))
heater_indices.from_numpy(np.array([
    0, 1, 1, 2, 2, 3, 3, 0,  # Bottom face
    4, 5, 5, 6, 6, 7, 7, 4,  # Top face
    0, 4, 1, 5, 2, 6, 3, 7   # Side edges
]))

drain_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(4,))
drain_vertices[0] = [xmin, -drain_size, -drain_size]
drain_vertices[1] = [xmin, drain_size, -drain_size]
drain_vertices[2] = [xmin, drain_size, drain_size]
drain_vertices[3] = [xmin, -drain_size, drain_size]

drain_indices = ti.field(dtype=ti.i32, shape=(8,))
drain_indices.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0]))

dt = 1e-5
n = 1000
R = 8.31
ball_radius = 2e-3  # 1.5e-3
ball_center = ti.Vector.field(3, dtype=ti.f32, shape=(n,))
ball_color = ti.Vector.field(3, dtype=ti.f32, shape=(n,))
black = (0.0, 0.0, 0.0)
T_set = 300
T_actual = None
m = 0.032
Vrms = math.sqrt(3 * R * T_set / m)

heater_E = 1.5 * R * 400

pos = np.random.uniform(low=-box_size, high=box_size, size=(n, 3))

v_error_ratio = 1e-3
v_abs = np.random.uniform(low=(1 - v_error_ratio)*Vrms, high=(1 + v_error_ratio)*Vrms, size=(n, 1))
theta = np.random.uniform(high=2*np.pi, size=(n, 1))
phi = np.random.uniform(high=2*np.pi, size=(n, 1))
vx = v_abs * np.cos(phi) * np.cos(theta)
vy = v_abs * np.cos(phi) * np.sin(theta)
vz = v_abs * np.sin(phi)
v = np.hstack((vx, vy, vz))

a = np.array([0.0, -9.8, 0.0])

pos = pos.astype(np.float32)
ball_center.from_numpy(pos)

blue = np.array([[0.0, 0.0, 0.545]])
red = np.array([[0.6, 0.0, 0.0]])

#Enlarge the window
root = tk.Tk()
root.withdraw()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

window = ti.ui.Window("Gas Simulation", (int(width/2.5), int(height/2.5)), vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

show_drain = False
inject_particles = False
add_heater = True
shake = True
init_diffusion_particles = False
injection_rate = dt*10
elapsed_time = 0
iter = 0
V_avg = 0.0
P_avg = 0.0
T_avg = 0.0
n_avg = 0.0
num_emit = 0
num_emit_total = 0
num_data = 0

# file = open("data/pt_n1000_v16.txt", "w")
file_tmp = open("data/temp-time_vshake_400.txt", "w")

#count = 0
while window.running:
    scene.lines(vertices=line_vertices, width=0.5, indices=line_indices, color=black)

    iter += 1

    # Visualize heater
    if add_heater:
        scene.lines(vertices=heater_vertices, width=0.5, indices=heater_indices, color=(1, 0, 0))  # Heater color

    if shake:
        #Use sine function to represent the movement of a piston in y-axis
        ymax = box_size * (1 + 0.8 * math.sin(elapsed_time * 300))
        line_vertices[0] = [xmin, ymax, zmin]
        line_vertices[1] = [xmax, ymax, zmin]
        line_vertices[2] = [xmax, ymax, zmax]
        line_vertices[3] = [xmin, ymax, zmax]

    # Initialize diffusion testing by adding bulk particles in one part of the box
    if init_diffusion_particles:
        num_diffus_particles = 10
        pos, v = add_diffusion_particles(pos, v, box_size, R, T_actual, m, num_diffus_particles)

        n += num_diffus_particles
        ball_center = ti.Vector.field(3, dtype=ti.f32, shape=(n,))
        ball_color = ti.Vector.field(3, dtype=ti.f32, shape=(n,))

        init_diffusion_particles = False

    # Update particle position and velocity
    pos, v = particle_motion(pos, v, a, dt)

    # collision between particles
    v = particle_collision(pos, v, ball_radius)

    if show_drain:
        scene.lines(vertices=drain_vertices, width=0.5, indices=drain_indices, color=black)
        emit_idx = emit_from_drain(pos, v, xmin, drain_size)
        # print(f"number of emitting particles: {emit_idx.shape}")
        if emit_idx.shape[0] > 0:
            num_emit += emit_idx.shape[0]
            n -= emit_idx.shape[0]
            ball_center = ti.Vector.field(3, dtype=ti.f32, shape=(n,))
            ball_color = ti.Vector.field(3, dtype=ti.f32, shape=(n,))
            pos = np.delete(pos, emit_idx, axis=0)
            v = np.delete(v, emit_idx, axis=0)

    n_avg = n_avg * (iter - 1)/iter + n / iter

    # collision between particles and the border of the box
    pos, v, P = border_collisions(pos, v, m, dt, xmin, xmax, ymin, ymax, zmin, zmax, add_heater, heater_xmin, heater_xmax, heater_zmin, heater_zmax, heater_E)
    pos = pos.astype(np.float32)
    ball_center.from_numpy(pos)

    # updates the color of particles according to the velocity
    v_abs = get_v_abs(v)
    clr = blue * (1.3 * Vrms - v_abs) + red * (v_abs - Vrms)
    clr = np.clip(clr, 0, 1)
    clr = clr.astype(np.float32)
    ball_color.from_numpy(clr)

    P_avg = P_avg * (iter - 1)/iter + P / iter

    V = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    V_avg = V_avg * (iter - 1)/iter + V / iter

    T_actual = cal_temperature(m, R, v)
    T_avg = T_avg * (iter - 1)/iter + T_actual / iter

    # get P-N data points
    # if iter % 150 == 0:
        # print(f"PV / nRT = {(P_avg * V_avg) / (n_avg * R * T_avg)}")
        # print(f"T = {T_avg}, P = {P_avg}, V = {V_avg}, P/n = {P_avg * V_avg}")
        # # file.write(f"{P_avg}, {n_avg}\n")
        # num_data += 1
        # if show_drain:
        #     print(f"{num_emit} particles emit.")
        #     num_emit_total += num_emit
        #     num_emit = 0
        
        # if num_data == 300:
        #     break

    if iter == 300:
        # init_diffusion_particles = True
        print(f"PV / nRT = {(P_avg * V_avg) / (n_avg * R * T_avg)}")
        print(f"P = {P_avg}, T = {T_avg} P/T = {P_avg / T_avg}")
        # file.write(f"{P_avg}, {T_avg}\n")
        file_tmp.write(f"{elapsed_time}, {T_avg}, {V_avg}\n")
        num_data += 1
        if show_drain:
            print(f"{num_emit} particles emit.")
            num_emit_total += num_emit
            num_emit = 0
        
        if T_avg >= 395 or num_data == 50:
            print(f"End at iteration = {num_data}, final temperature is {T_avg}")
            break
        
        # adjust volume to do the P-V analysis
        # ymax = box_size * (0.25 + 0.05 * num_data)
        # ymin = -box_size * (0.25 + 0.05 * num_data)
        # line_vertices[0] = [xmin, ymax, zmin]
        # line_vertices[1] = [xmax, ymax, zmin]
        # line_vertices[2] = [xmax, ymax, zmax]
        # line_vertices[3] = [xmin, ymax, zmax]
        # line_vertices[4] = [xmin, ymin, zmin]
        # line_vertices[5] = [xmax, ymin, zmin]
        # line_vertices[6] = [xmax, ymin, zmax]
        # line_vertices[7] = [xmin, ymin, zmax]
        P_avg = 0.0
        V_avg = 0.0
        n_avg = 0.0
        T_avg = 0.0
        iter = 0

    camera.position(2.0, 2.0, 4.0)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, per_vertex_color=ball_color)

    canvas.scene(scene)
    window.show()

    elapsed_time += dt

# file.close()
file_tmp.close()
