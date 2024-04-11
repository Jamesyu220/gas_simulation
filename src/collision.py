# Update speed function
def update_velocity(v1, v2, pos1, pos2):
    norm_vector = (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
    v1_new = v1 - np.dot(v1 - v2, norm_vector) * norm_vector
    v2_new = v2 - np.dot(v2 - v1, norm_vector) * norm_vector
    return v1_new, v2_new

# Calculate the index of the bucket the particle is in
def get_bucket_idx(pos, box_size, bucket_size):
    return tuple(((pos + box_size) // bucket_size).astype(int))

# Create buckets and assign particles
def create_buckets(x, box_size, bucket_size):
    buckets = {}
    for i, pos in enumerate(x):
        idx = get_bucket_idx(pos, box_size, bucket_size)
        if idx not in buckets:
            buckets[idx] = []
        buckets[idx].append(i)
    return buckets

# Handle collisions
def particle_collision(x, v, box_size, ball_radius):
    bucket_size = 2 * ball_radius * 2
    buckets = create_buckets(x, box_size, bucket_size)
    for particles in buckets.values():
        if len(particles) < 2:
            continue  # Skip buckets with 0 or 1 particle
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                pi, pj = particles[i], particles[j]
                pos1, pos2 = x[pi], x[pj]
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 2 * ball_radius:
                    v[pi], v[pj] = update_velocity(v[pi], v[pj], pos1, pos2)
    return v
