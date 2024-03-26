def get_bucket_idx(pos, bucket_size, box_size):
    # Calculate the index of the bucket where the particle is located
    return tuple(int((p + box_size) // bucket_size) for p in pos)

def create_buckets(x, bucket_size, box_size):
    buckets = {}
    for i, pos in enumerate(x):
        idx = get_bucket_idx(pos, bucket_size, box_size)
        if idx not in buckets:
            buckets[idx] = []
        buckets[idx].append(i)
    return buckets

def check_collision(pos1, pos2, ball_radius):
    # Calculate whether the distance between two particles is less than twice the radius
    return np.linalg.norm(pos1 - pos2) < 2 * ball_radius

def update_velocity(v1, v2, pos1, pos2):
    # Simplified post-collision velocity update logic
    norm_vector = (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
    v1_new = v1 - np.dot(v1 - v2, norm_vector) * norm_vector
    v2_new = v2 - np.dot(v2 - v1, norm_vector) * norm_vector
    return v1_new, v2_new

def get_neighbor_buckets(bucket_idx):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue  # skip self
                neighbor_idx = (bucket_idx[0] + dx, bucket_idx[1] + dy, bucket_idx[2] + dz)
                neighbors.append(neighbor_idx)
    return neighbors


def particle_collision(x, v, ball_radius, bucket_size, box_size):
    n = len(x)
    buckets = create_buckets(x, bucket_size, box_size)
    
    # Iterate through each bucket
    for bucket_idx, particles in buckets.items():
        # Check particle collisions within the current bucket
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                if check_collision(x[particles[i]], x[particles[j]], ball_radius):
                    v[particles[i]], v[particles[j]] = update_velocity(v[particles[i]], v[particles[j]], x[particles[i]], x[particles[j]])
                    
        # Check for collisions with adjacent buckets
        neighbors = get_neighbor_buckets(bucket_idx)
        for neighbor_idx in neighbors:
            if neighbor_idx in buckets:  # If adjacent buckets exist
                for i in particles:
                    for j in buckets[neighbor_idx]:
                        if check_collision(x[i], x[j], ball_radius):
                            v[i], v[j] = update_velocity(v[i], v[j], x[i], x[j])
    return v
