# import taichi as ti
import numpy as np
import math

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

    return pos, v
    