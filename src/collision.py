def particle_collision(x, v, ball_radius):
    """
    Updates the velocity after a fully elastic collision between particles.

    parameter:
    - x: Position array of particles, size (n, 3), where n is the number of particles.
    - v: velocity array of particles, same size as x.
    - ball_radius: radius of the particle.

    return:
    - v: updated velocity array.
    """
    n = len(x)  # number of particles
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(x[i] - x[j])
            if distance < 2 * ball_radius:
                # unit vector of collision direction
                norm_vector = (x[j] - x[i]) / distance
                # Calculates velocity updates based on conservation of momentum and kinetic energy
                v_i_new = v[i] - np.dot(v[i] - v[j], norm_vector) * norm_vector
                v_j_new = v[j] - np.dot(v[j] - v[i], norm_vector) * norm_vector
                v[i], v[j] = v_i_new, v_j_new
    return v
