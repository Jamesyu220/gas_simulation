import taichi as ti
import numpy as np
from ideal_gas import box_size
ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu), ti.init(arch=ti.vulkan)

#assume rectangular box
boundary_x = [0, box_size] #lower + upper x coords for box
boundary_y = [0, box_size] #lower + upper y coords for box
boundary_z = [0, box_size] #lower + upper z coords for box

# @ti.kernel
def boundary_check(x, y, z)->int:
    """
    Checks that the particle is not hitting the boundary wall
    x, y: ball center coords
    ball_radius: global size of a particle
    returns 0 if no boundary collision
    returns 1 if collision with x bounds only
    returns 2 if collision with y bounds only
    returns 3 if collision with z bounds only
    returns 4 if collision with x & y bounds (corner)
    returns 5 if collision with x & z bounds (corner)
    returns 6 if collision with y & z bounds (corner)
    returns 7 if collision with x, y & z bounds (corner)

    """
    global ball_radius, boundary_x, boundary_y
    hitsXBounds = (x - ball_radius <= boundary_x[0] or boundary_x[1] <= x + ball_radius)
    hitsYBounds = (y - ball_radius <= boundary_y[0] or boundary_y[1] <= y + ball_radius)
    hitsZBounds = (z - ball_radius <= boundary_z[0] or boundary_z[1] <= z + ball_radius)

    if hitsXBounds & hitsYBounds & hitsZBounds:
        return 7
    if hitsYBounds & hitsZBounds:
        return 6  
    if hitsXBounds & hitsZBounds:
        return 5
    if hitsXBounds & hitsYBounds:
        return 4 
    elif hitsXBounds:
        return 1
    elif hitsYBounds:
        return 2
    elif hitsZBounds:
        return 3

    return 0


# @ti.kernel
def border_collision(v_x, v_y, v_z, x, y, z, heat_factor):
    """
    Runs every step of d_t
    v_x, v_y: velocity components
    x, y: location of the center of mass of the particle
    heat_factor: if the container is heated, 
    adds energy in the form of a coefficient of velocity to the particle
    """
    hitsBounds = boundary_check(x, y, z)
    if hitsBounds == 1:  
        #increase velocity in only the direction the particle is being reflected    
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.dot([v_y, v_z], [v_y, v_z])) / ti.math.pow(v_x, 2) 
        #reverse x velocity; 
        v_x = -1 * heat_factor * v_x
    elif hitsBounds == 2:
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.dot([v_x, v_z], [v_x, v_z])) / ti.math.pow(v_y, 2) 
        #reverse y velocity; 
        v_y = -1 * heat_factor *  v_y
    elif hitsBounds == 3:
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.dot([v_x, v_y], [v_x, v_y])) / ti.math.pow(v_z, 2) 
        #reverse y velocity; 
        v_z = -1 * heat_factor *  v_z
    elif hitsBounds == 4:
        #increase velocity in only the 2 directions the particle is being reflected    
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.pow(v_z, 2)) / ti.math.dot([v_x, v_y], [v_x, v_y]) 
        
        #reverse both velocities
        v_x = -1 * heat_factor * v_x
        v_y = -1 * heat_factor * v_y
    elif hitsBounds == 5:
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.pow(v_y, 2)) / ti.math.dot([v_x, v_z], [v_x, v_z]) 
        #reverse both velocities
        v_x = -1 * heat_factor * v_x
        v_z = -1 * heat_factor * v_z
    elif hitsBounds == 6:
        heat_factor = (heat_factor * ti.math.dot([v_x, v_y, v_z], [v_x, v_y, v_z]) \
                    - ti.math.pow(v_x, 2)) / ti.math.dot([v_y, v_z], [v_y, v_z]) 
        #reverse both velocities
        v_z = -1 * heat_factor * v_z
        v_y = -1 * heat_factor * v_y
    elif hitsBounds == 7:
        #since the increase in velocity is being applied to two components
        #use the square root as a coefficient 
        heat_factor = ti.math.sqrt(heat_factor)
        #reverse both velocities
        v_x = -1 * heat_factor * v_x
        v_y = -1 * heat_factor * v_y
        v_z = -1 * heat_factor * v_z

    return v_x, v_y, v_z
