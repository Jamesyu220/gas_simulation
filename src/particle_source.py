# import taichi as ti
import numpy as np
import math
# ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)


# @ti.kernel
def add_particle(x, v, a, dt, box_size, ball_radius, R, m, T):
    Vrms = math.sqrt(3 * R * T / m) 

    # Inject the particles just below the boundary so they don't get stuck.
    source_coords = np.array([[0,box_size-(ball_radius+.001),0]])

    # # Start with a simple perfectly straight injection of particles. I will update this with variance.
    # particle_v = np.array([[0, -Vrms, 0]])

    # We want the particles to have some variance in their injection trajectory.
    # Start by choosing a random velocity in the injection direction/axes and then split the remaining
    # velocity magnitude between the other two directions/axes.
    vy = np.random.uniform(-Vrms*0.9, -Vrms*0.7) # Bounds are arbitrarily chosen.
    temp_v = Vrms - abs(vy)
    vx = np.random.uniform(-temp_v, temp_v)
    temp_v -= vx
    sign = np.random.choice([-1, 1])
    vz = sign*temp_v
    
    particle_v = np.array([[vx, vy, vz]])
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