import taichi as ti
import numpy as np
import time
import math
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

from particles import particle_motion
from particle_source import add_particle

box_size = 0.8
drain_size = 0.1

line_vertices = ti.Vector.field(3, dtype=float, shape=(8,))
line_vertices[0] = [-box_size, box_size, -box_size]
line_vertices[1] = [box_size, box_size, -box_size]
line_vertices[2] = [box_size, box_size, box_size]
line_vertices[3] = [-box_size, box_size, box_size]
line_vertices[4] = [-box_size, -box_size, -box_size]
line_vertices[5] = [box_size, -box_size, -box_size]
line_vertices[6] = [box_size, -box_size, box_size]
line_vertices[7] = [-box_size, -box_size, box_size]

line_indices = ti.field(dtype=ti.i32, shape=(24,))
line_indices.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4]))


drain_vertices = ti.Vector.field(3, dtype=float, shape=(4,))
drain_vertices[0] = [-drain_size, -box_size, -drain_size]
drain_vertices[1] = [drain_size, -box_size, -drain_size]
drain_vertices[2] = [drain_size, -box_size, drain_size]
drain_vertices[3] = [-drain_size, -box_size, drain_size]

drain_indices = ti.field(dtype=ti.i32, shape=(8,))
drain_indices.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0]))

dt = 5e-6
n = 5000
R = 8.31
ball_radius = 1.5e-3
ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
ball_color = ti.Vector.field(3, dtype=float, shape=(n,))
black = (0.0, 0.0, 0.0)
T = 300
m = 0.032
Vrms = math.sqrt(3 * R * T / m)

clr_ori = np.array([0.0, 0.0, 0.545])
clr_ins = np.array([1.0, 0.0, 0.0])
ball_color.from_numpy(np.vstack((np.tile(clr_ori, (3750, 1)), np.tile(clr_ins, (1250, 1)))))

x = np.random.uniform(low=-box_size, high=box_size, size=(n, 3))

v_abs = np.random.uniform(low=0.9*Vrms, high=1.1*Vrms, size=(n, 1))
theta = np.random.uniform(high=2*np.pi, size=(n, 1))
phi = np.random.uniform(high=2*np.pi, size=(n, 1))
vx = v_abs * np.cos(phi) * np.cos(theta)
vy = v_abs * np.cos(phi) * np.sin(theta)
vz = v_abs * np.sin(phi)
v = np.hstack((vx, vy, vz))

a = np.array([0.0, -9.8, 0.0])

ball_center.from_numpy(x)

window = ti.ui.Window("Taichi Gascd  Simulation on GGUI", (480, 320),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

show_drain = False
inject_particles = False
injection_rate = dt*10
elapsed_time = 0

#count = 0
while window.running:
    scene.lines(vertices=line_vertices, width=0.5, indices=line_indices, color=black)

    if show_drain:
        scene.lines(vertices=drain_vertices, width=0.5, indices=drain_indices, color=black)

    # timeStamp = time.time()
    # scale = box_size * (1 + math.sin(timeStamp))
    # line_vertices[0] = [-box_size, box_size * scale, -box_size]
    # line_vertices[1] = [box_size, box_size * scale, -box_size]
    # line_vertices[2] = [box_size, box_size * scale, box_size]
    # line_vertices[3] = [-box_size, box_size * scale, box_size]

    # Update particle position and velocity
    x, v = particle_motion(x, v, a, dt, box_size, ball_radius)
    ball_center.from_numpy(x)
    
    # Inject particles
    if inject_particles and (elapsed_time >= injection_rate):
        x, v = add_particle(x, v, a, dt, box_size, ball_radius, R, m, T)
        n += 1
        ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
        ball_center.from_numpy(x)
        elapsed_time = 0

    camera.position(2.0, 2.0, 4.0)
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