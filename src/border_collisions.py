import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

#assume rectangular box
boundary_x = [0, 100] #lower + upper x coords for box
boundary_y = [0, 100] #lower + upper y coords for box
# @ti.kernel
def boundary_check(x, y)->int:
    """
    Checks that the particle is not hitting the boundary wall
    x, y: ball center coords
    ball_radius: global size of a particle
    returns 0 if no boundary collision
    returns 1 if collision with horizontal walls
    returns 2 if collision with vertical walls
    returns 3 if collision with both walls (in the case of a corner)
    """
    global ball_radius, boundary_x, boundary_y
    hitsXBounds = (x - ball_radius <= boundary_x[0] or boundary_x[1] <= x + ball_radius)
    hitsYBounds = (y - ball_radius <= boundary_y[0] or boundary_y[1] <= y + ball_radius)

    if hitsXBounds & hitsYBounds:
        return 3
    elif hitsXBounds:
        return 1
    elif hitsYBounds:
        return 2

    return 0


# @ti.kernel
def border_collision(v_x, v_y, x, y, heat_factor):
    """
    Runs every step of d_t
    v_x, v_y: velocity components
    x, y: location of the center of mass of the particle
    heat_factor: if the container is heated, 
    adds energy in the form of a velocity multiplier to the particle
    """
    hitsBounds = boundary_check(x, y)
    if hitsBounds == 1:
        #reverse x velocity; 
        v_x = -1 * heat_factor * v_x
    elif hitsBounds == 2:
        #reverse y velocity; 
        v_y = -1 * heat_factor *  v_y
    elif hitsBounds == 3:
        #since the increase in velocity is being applied to two components
        #use the square root as a coefficient 
        heat_factor = ti.math.sqrt(heat_factor)
        #reverse both velocities
        v_x = -1 * heat_factor * v_x
        v_y = -1 * heat_factor * v_y
    return v_x, v_y
