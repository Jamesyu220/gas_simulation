# @ti.kernel
def particle_motion(x, v, a, dt, box_size, ball_radius):
    v = v + a * dt
    boundary_condition = (x <= -box_size + ball_radius) | (x >= box_size - ball_radius)
    v[boundary_condition] = v[boundary_condition] * (-1)
    x = x + v * dt

    return x, v