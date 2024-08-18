
import sys

class AABB:

    def __init__(self, x_min: float, y_min: float, z_min: float, x_max: float, y_max: float, z_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def is_empty(self):
        """" Returns true if the AABB is degenerated. """
        return self.x_min >= self.x_max or self.y_min >= self.y_max or self.z_min >= self.z_max
    
    def copy(self):
        return AABB(self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max)

# Returns the AABB which encloses all vertices given in the argument.
def get_AABB_from_vertices(vertices):

    aabb = AABB(sys.float_info.max, sys.float_info.max, sys.float_info.max, 
                sys.float_info.min, sys.float_info.min, sys.float_info.min)

    for v in vertices:
        if v[0] < aabb.x_min:
            aabb.x_min = v[0]
        if v[0] > aabb.x_max:
            aabb.x_max = v[0]
        if v[1] < aabb.y_min:
            aabb.y_min = v[1]
        if v[1] > aabb.y_max:
            aabb.y_max = v[1]
        if v[2] < aabb.z_min:
            aabb.z_min = v[2]
        if v[2] > aabb.z_max:
            aabb.z_max = v[2]
        
    return aabb

# Returns the AABB which encloses all primitives given in the argument.
def get_AABB_from_primitives(primitves):

    if len(primitves) == 0:
        return AABB(0,0,0,0,0,0)

    vertices = []

    for p in primitves:
        vertices.append(p[0])
        vertices.append(p[1])
        vertices.append(p[2])

    return get_AABB_from_vertices(vertices)
