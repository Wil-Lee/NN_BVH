
import sys
import nn_types

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
    
    def get_min(self, axis: nn_types.Axis):
        if axis == nn_types.Axis.x:
            return self.x_min
        if axis == nn_types.Axis.y:
            return self.y_min
        if axis == nn_types.Axis.z:
            return self.y_min
        return - 1
    
    def get_max(self, axis: nn_types.Axis):
        if axis == nn_types.Axis.x:
            return self.x_max
        if axis == nn_types.Axis.y:
            return self.y_max
        if axis == nn_types.Axis.z:
            return self.y_max
        return - 1
    
    def set_min(self, axis: nn_types.Axis, new_min: float):
        if axis == nn_types.Axis.x:
            self.x_min = new_min
        if axis == nn_types.Axis.y:
            self.y_min = new_min
        if axis == nn_types.Axis.z:
            self.z_min = new_min
    
    def set_max(self, axis: nn_types.Axis, new_max: float):
        if axis == nn_types.Axis.x:
            self.x_max = new_max
        if axis == nn_types.Axis.y:
            self.y_max = new_max
        if axis == nn_types.Axis.z:
            self.z_max = new_max

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
