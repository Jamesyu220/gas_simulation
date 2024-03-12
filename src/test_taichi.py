import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)


dt = 1e-4
d = 9
ball_radius = 0.005
ball_center = ti.Vector.field(3, dtype=float, shape=(d**2,))

x = np.zeros((d, d, 3))
v = np.zeros((d, d, 3))
a = np.zeros((d, d, 3))

# @ti.kernel
def init_para():
    global x, v, a, ball_center
    x1 = (-(d // 2) + np.arange(d**2) // d) * 0.2
    x2 = (-(d // 2) + np.arange(d**2) % d) * 0.2
    x3 = np.zeros(d**2)
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    x3 = x3.reshape((-1, 1))
    x = np.concatenate((x1, x2, x3), axis=1)
    x = x.reshape((d, d, 3))
    v = np.random.uniform(size=x.shape) * 5.0
    a = np.array([0.0, -9.8, 0.0])
    a = a.reshape((1, 1, 3))

    flatten_x = np.reshape(x, (d**2, 3))
    ball_center.from_numpy(flatten_x)

# @ti.kernel
def particle_motion():
    global x, v, a, ball_center
    v = v + a * dt
    boundary_condition = (x <= -0.2 * (d // 2) + ball_radius) | (x >= 0.2 * (d // 2) - ball_radius)
    v[boundary_condition] = v[boundary_condition] * (-1)
    x = x + v * dt

# @ti.kernel
def update_pos():
    global x, v, a, ball_center
    flatten_x = np.reshape(x, (d**2, 3))
    ball_center.from_numpy(flatten_x)

window = ti.ui.Window("Taichi Gascd  Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
init_para()

while window.running:
    if current_t > 1.5:
        # Reset
        init_para()
        current_t = 0

    
    particle_motion()
    update_pos()
    current_t += dt

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()