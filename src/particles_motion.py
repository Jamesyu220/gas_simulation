import numpy as np
# @ti.kernel
def particle_motion(pos, v, a, dt):
    v = v + a * dt
    pos = pos + v * dt

    return pos, v

def border_collisions(pos, v, m, xmin, xmax, ymin, ymax, zmin, zmax):
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]
    v_x = v[:, 0]
    v_y = v[:, 1]
    v_z = v[:, 2]

    J = 0
    # bc stands for boundary condition
    bc_x_min = (pos_x <= xmin)
    pos[bc_x_min, 0] = xmin
    bc_x_min = bc_x_min & (v_x < 0)
    v[bc_x_min, 0] *= -1
    J += 2 * m * np.sum(v[bc_x_min, 0]).item()

    bc_x_max = (pos_x >= xmax)
    pos[bc_x_max, 0] = xmax
    bc_x_max = bc_x_max & (v_x > 0)
    v[bc_x_max, 0] *= -1
    J += -2 * m * np.sum(v[bc_x_max, 0]).item()

    bc_y_min = (pos_y <= ymin)
    pos[bc_y_min, 1] = ymin
    bc_y_min = bc_y_min & (v_y < 0)
    v[bc_y_min, 1] *= -1
    J += 2 * m * np.sum(v[bc_y_min, 1]).item()

    bc_y_max = (pos_y >= ymax)
    pos[bc_y_max, 1] = ymax
    bc_y_max = bc_y_max & (v_y > 0)
    v[bc_y_max, 1] *= -1
    J += -2 * m * np.sum(v[bc_y_max, 1]).item()

    bc_z_min = (pos_z <= zmin)
    pos[bc_z_min, 2] = zmin
    bc_z_min = bc_z_min & (v_z < 0)
    v[bc_z_min, 2] *= -1
    J += 2 * m * np.sum(v[bc_z_min, 2]).item()

    bc_z_max = (pos_z >= zmax)
    pos[bc_z_max, 2] = zmax
    bc_z_max = bc_z_max & (v_z > 0)
    v[bc_z_max, 2] *= -1
    pos[bc_z_max, 2] = zmax
    J += -2 * m * np.sum(v[bc_z_max, 2]).item()

    return pos, v, J