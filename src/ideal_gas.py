import taichi as ti
import numpy as np
import time
import math
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

from particles_motion import particle_motion, border_collisions
from particle_source import add_particle, add_diffusion_particles
from temperature import get_v_abs, cal_temperature

box_size = 0.8
drain_size = 0.1
xmin = -box_size
xmax = box_size
ymin = -box_size
ymax = box_size
zmin = -box_size
zmax = box_size

line_vertices = ti.Vector.field(3, dtype=float, shape=(8,))
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


drain_vertices = ti.Vector.field(3, dtype=float, shape=(4,))
drain_vertices[0] = [xmin, -drain_size, -drain_size]
drain_vertices[1] = [xmin, drain_size, -drain_size]
drain_vertices[2] = [xmin, drain_size, drain_size]
drain_vertices[3] = [xmin, -drain_size, drain_size]

drain_indices = ti.field(dtype=ti.i32, shape=(8,))
drain_indices.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0]))

dt = 5e-6
n = 500#5000
R = 8.31
ball_radius = 1.5e-2 #1.5e-3
ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
ball_color = ti.Vector.field(3, dtype=float, shape=(n,))
black = (0.0, 0.0, 0.0)
T_set = 300
T_actual = None
m = 0.032
Vrms = math.sqrt(3 * R * T_set / m)

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

ball_center.from_numpy(pos)

blue = np.array([[0.0, 0.0, 0.545]])
red = np.array([[0.6, 0.0, 0.0]])

window = ti.ui.Window("Taichi Gascd  Simulation on GGUI", (480, 320),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

# Set toggles
show_drain = True
increase_volume = False
inject_particles = False
init_diffusion_particles = True

injection_rate = dt*10
elapsed_time = 0
iter = 0
V_avg = (2 * box_size) ** 3
P_avg = 0.0
T_avg = 0.0

#count = 0
while window.running:
    scene.lines(vertices=line_vertices, width=0.5, indices=line_indices, color=black)

    if show_drain:
        scene.lines(vertices=drain_vertices, width=0.5, indices=drain_indices, color=black)

    iter += 1
    # timeStamp = time.time()
    if increase_volume:
        ymax = box_size * (1 + 0.5 * math.sin(elapsed_time * 300))
        line_vertices[0] = [xmin, ymax, zmin]
        line_vertices[1] = [xmax, ymax, zmin]
        line_vertices[2] = [xmax, ymax, zmax]
        line_vertices[3] = [xmin, ymax, zmax]

    # Update particle position and velocity
    pos, v = particle_motion(pos, v, a, dt)

    pos, v, P = border_collisions(pos, v, m, dt, xmin, xmax, ymin, ymax, zmin, zmax)
    ball_center.from_numpy(pos)

    v_abs = get_v_abs(v)
    clr = blue * (1.3 * Vrms - v_abs) + red * (v_abs - Vrms)
    clr = np.clip(clr, 0, 1)
    ball_color.from_numpy(clr)

    P_avg = P_avg * (iter - 1)/iter + P / iter

    V = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    T_actual = cal_temperature(m, R, v)
    T_avg = T_avg * (iter - 1)/iter + T_actual / iter

    # if abs((P * V) / (n * R * T_actual) - 1.0) > 0.05:
    #     print(f"Big error: PV/nRT = {(P * V) / (n * R * T_actual)}")

    if iter % 100 == 0:
        print(f"PV / nRT = {(P_avg * V_avg) / (n * R * T_avg)}")
    
    # Inject particles
    if inject_particles and (elapsed_time >= injection_rate):
        pos, v = add_particle(pos, v, a, dt, box_size, ball_radius, R, m, T_set)
        n += 1
        ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
        ball_center.from_numpy(pos)
        elapsed_time = 0

    # Initialize diffusion testing by adding bulk particles in one part of the box
    if init_diffusion_particles:
        num_diffus_particles = 100
        pos, v = add_diffusion_particles(pos, v, a, dt, box_size, ball_radius, R, m, T_set, num_diffus_particles)

        n += num_diffus_particles
        ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
        ball_center.from_numpy(pos)

        init_diffusion_particles = False



    #camera.position(2.0, 2.0, 4.0)
    camera.position(-2.0, -2.0, 4.0)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, per_vertex_color=ball_color)
    #count += 1
    # test_particle = ti.Vector.field(3, dtype=float, shape=(1,))
    # test_particle.from_numpy(np.array([[0,box_size,0]]))
    # scene.particles(test_particle, radius=ball_radius*5, color=(0.0, 0.0, 0.0))
    
    canvas.scene(scene)
    window.show()

    elapsed_time += dt