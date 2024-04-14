import numpy as np
from src.temperature import get_v_abs
# @ti.kernel
def particle_motion(pos, v, a, dt):
    v = v + a * dt
    pos = pos + v * dt

    return pos, v

def particle_collision(pos, v, ball_radius):
    pos_x = pos[:, 0:1]
    pos_y = pos[:, 1:2]
    pos_z = pos[:, 2:3]

    delta_x = np.absolute(pos_x - pos_x.T)
    delta_y = np.absolute(pos_y - pos_y.T)
    delta_z = np.absolute(pos_z - pos_z.T)

    # dist = np.square(delta_x) + np.square(delta_y) + np.square(delta_z)
    # min_dist = np.min(dist, axis=1)
    # closest_neighbor = np.argmin(dist, axis=1)
    collision_condition = (delta_x < 2 * ball_radius) & (delta_y < 2 * ball_radius) & (delta_z < 2 * ball_radius)
    collision_neighbor = np.argwhere(collision_condition)
    _, idx = np.unique(collision_neighbor[:, 0], return_index=True)
    collision_neighbor = collision_neighbor[idx, :]
    v[collision_neighbor[:, 0], :] = v[collision_neighbor[:, 1], :]

    return v


def border_collisions(pos, v, m, dt, xmin, xmax, ymin, ymax, zmin, zmax, add_heater, heater_xmin, heater_xmax, heater_zmin, heater_zmax, heater_E):
    # Bottom area where the heater affects particles

    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]
    v_x = v[:, 0]
    v_y = v[:, 1]
    v_z = v[:, 2]

    # Detect particles hitting the bottom area above the heater
    # bc_y_min_heater = (pos_y <= ymin) & (pos_x >= heater_xmin) & (pos_x <= heater_xmax) & (pos_z >= heater_zmin) & (pos_z <= heater_zmax)
    # pos[bc_y_min_heater, 1] = ymin
    # bc_y_min_heater = bc_y_min_heater & (v_y < 0)

    # # Update velocity for particles hitting the heater area
    # v[bc_y_min_heater, 1] *= -1
    # v_abs = get_v_abs(v[bc_y_min_heater])
    # v[bc_y_min_heater] *= np.sqrt(1 + delta_E / (m * v_abs ** 2))

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
    orginal_vy = v[bc_y_min, 1]
    v[bc_y_min, 1] *= -1
    if add_heater:
        bc_heater = bc_y_min & (pos_x >= heater_xmin) & (pos_x <= heater_xmax) & (pos_z >= heater_zmin) & (pos_z <= heater_zmax)
        new_Ey = heater_E - 0.5 * m * np.square(v[bc_heater, 0]) - 0.5 * m * np.square(v[bc_heater, 2])
        v[bc_heater, 1] = np.sqrt(2 * new_Ey / m)
    J3 = m * np.sum(v[bc_y_min, 1] - orginal_vy).item()
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