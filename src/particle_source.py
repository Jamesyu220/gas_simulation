# import taichi as ti
import numpy as np
import math


# @ti.kernel
# def add_particle(pos, v, a, dt, box_size, ball_radius, R, m, T):
#     Vrms = math.sqrt(3 * R * T / m) 

#     # Inject the particles just below the boundary so they don't get stuck.
#     source_coords = np.array([[0,box_size-(ball_radius+.001),0]])

#     # # Start with a simple perfectly straight injection of particles. I will update this with variance.
#     # particle_v = np.array([[0, -Vrms, 0]])

#     # We want the particles to have some variance in their injection trajectory.
#     # Start by choosing a random velocity in the injection direction/axes and then split the remaining
#     # velocity magnitude between the other two directions/axes.
#     vy = np.random.uniform(-Vrms*0.9, -Vrms*0.7) # Bounds are arbitrarily chosen.
#     temp_v = Vrms - abs(vy)
#     vx = np.random.uniform(-temp_v, temp_v)
#     temp_v -= vx
#     sign = np.random.choice([-1, 1])
#     vz = sign*temp_v
    
#     particle_v = np.array([[vx, vy, vz]])
#     #particle_v = np.random.uniform(low=0.9*Vrms, high=1.1*Vrms, size=(1,3))
    
#     # Add particle to particle vector
#     pos = np.append(pos, source_coords, axis=0)
#     v = np.append(v, particle_v, axis=0)

#     return pos, v

def add_diffusion_particles(pos, v, box_size, R, T, m, n_added):
    pos_diffus = np.random.uniform(low=-box_size, high=-box_size/2, size=(n_added, 3))
    v_abs = np.full((n_added, 1), math.sqrt(3 * R * T / m))

    theta = np.random.uniform(high=2*np.pi, size=(n_added, 1))
    phi = np.random.uniform(high=2*np.pi, size=(n_added, 1))
    vx = v_abs * np.cos(phi) * np.cos(theta)
    vy = v_abs * np.cos(phi) * np.sin(theta)
    vz = v_abs * np.sin(phi)
    v_diffus = np.hstack((vx, vy, vz))

    pos = np.append(pos, pos_diffus, axis=0)
    v = np.append(v, v_diffus, axis=0)

    # print(f"new pos: {pos_diffus}")
    # print(np.shape(pos))
    # print(f"new v: {v_diffus}")
    # print(np.shape(v))

    return pos, v
    