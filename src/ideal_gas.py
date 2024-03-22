import taichi as ti
import numpy as np
import math
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

from particles import particle_motion

dt = 1e-5
n = 10000
R = 8.31
ball_radius = 0.005
ball_center = ti.Vector.field(3, dtype=float, shape=(n,))
box_size = 0.8
drain_size = 0.1
black = (0.0, 0.0, 0.0)
T = 300
m = 0.032
Vrms = math.sqrt(3 * R * T / m)

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

x = np.random.uniform(low=-box_size, high=box_size, size=(n, 3))
v = np.random.uniform(low=0.9*Vrms, high=1.1*Vrms, size=x.shape)
a = np.array([0.0, -9.8, 0.0])
ball_center.from_numpy(x)

window = ti.ui.Window("Taichi Gascd  Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

show_drain = True

while window.running:
    scene.lines(vertices=line_vertices, width=0.5, indices=line_indices, color=black)

    if show_drain:
        scene.lines(vertices=drain_vertices, width=0.5, indices=drain_indices, color=black)

    x, v = particle_motion(x, v, a, dt, box_size, ball_radius)
    ball_center.from_numpy(x)

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()