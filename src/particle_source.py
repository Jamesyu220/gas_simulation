# import taichi as ti
import numpy as np
import math
# ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)


# @ti.kernel
def add_particle(x, v, a, dt, box_size, ball_radius, R, m, T):
    Vrms = math.sqrt(3 * R * T / m) 
    source_coords = np.array([[0,box_size,0]])

    # Start with a simple perfectly straight injection of particles. I will update this with variance.
    particle_v = np.array([[0, -Vrms, 0]])
    #particle_v = np.random.uniform(low=0.9*Vrms, high=1.1*Vrms, size=(1,3))
    
    # Add particle to particle vector
    x = np.append(x, source_coords, axis=0)
    v = np.append(v, particle_v, axis=0)

    return x, v


# def particle_motion(x, v, a, dt, box_size, ball_radius):
#     v = v + a * dt
#     boundary_condition = (x <= -box_size + ball_radius) | (x >= box_size - ball_radius)
#     v[boundary_condition] = v[boundary_condition] * (-1)
#     x = x + v * dt

#     return x, v