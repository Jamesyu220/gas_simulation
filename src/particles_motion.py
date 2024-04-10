import numpy as np
# @ti.kernel
def particle_motion(pos, v, a, dt):
    v = v + a * dt
    pos = pos + v * dt

    return pos, v

def border_collisions(pos, v, m, dt, xmin, xmax, ymin, ymax, zmin, zmax):
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]
    v_x = v[:, 0]
    v_y = v[:, 1]
    v_z = v[:, 2]

    # bc stands for boundary condition
    bc_x_min = (pos_x <= xmin)
    pos[bc_x_min, 0] = xmin
    bc_x_min = bc_x_min & (v_x < 0)
    v[bc_x_min, 0] *= -1
    J1 = 2 * m * np.sum(v[bc_x_min, 0]).item()
    F1 = J1 / dt
    P1 = F1 / ((ymax  - ymin) * (zmax - zmin))

    bc_x_max = (pos_x >= xmax)
    pos[bc_x_max, 0] = xmax
    bc_x_max = bc_x_max & (v_x > 0)
    v[bc_x_max, 0] *= -1
    J2 = -2 * m * np.sum(v[bc_x_max, 0]).item()
    F2 = J2 / dt
    P2 = F2 / ((ymax  - ymin) * (zmax - zmin))

    bc_y_min = (pos_y <= ymin)
    pos[bc_y_min, 1] = ymin
    bc_y_min = bc_y_min & (v_y < 0)
    v[bc_y_min, 1] *= -1
    J3 = 2 * m * np.sum(v[bc_y_min, 1]).item()
    F3 = J3 / dt
    P3 = F3 / ((xmax  - xmin) * (zmax - zmin))

    bc_y_max = (pos_y >= ymax)
    pos[bc_y_max, 1] = ymax
    bc_y_max = bc_y_max & (v_y > 0)
    v[bc_y_max, 1] *= -1
    J4 = -2 * m * np.sum(v[bc_y_max, 1]).item()
    F4 = J4 / dt
    P4 = F4 / ((xmax  - xmin) * (zmax - zmin))

    bc_z_min = (pos_z <= zmin)
    pos[bc_z_min, 2] = zmin
    bc_z_min = bc_z_min & (v_z < 0)
    v[bc_z_min, 2] *= -1
    J5 = 2 * m * np.sum(v[bc_z_min, 2]).item()
    F5 = J5 / dt
    P5 = F5 / ((xmax  - xmin) * (ymax - ymin))

    bc_z_max = (pos_z >= zmax)
    pos[bc_z_max, 2] = zmax
    bc_z_max = bc_z_max & (v_z > 0)
    v[bc_z_max, 2] *= -1
    pos[bc_z_max, 2] = zmax
    J6 = -2 * m * np.sum(v[bc_z_max, 2]).item()
    F6 = J6 / dt
    P6 = F6 / ((xmax  - xmin) * (ymax - ymin))

    P = (P1 + P2 + P3 + P4 + P5 + P6) / 6.0

    return pos, v, P

def emit_from_drain(pos, v, xmin, drain_size):
    drain_condition = (pos[:, 0] <= xmin) & (abs(pos[:, 1]) <= drain_size) & (abs(pos[:, 2]) <= drain_size) & (v[:, 0] < 0)
    # print(f"shape of drain condition: {drain_condition.shape}")
    return np.argwhere(drain_condition).flatten()