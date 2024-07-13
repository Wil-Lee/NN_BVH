from nn_AABB import *
from nn_types import *
import sys

class BVHNode:
    def __init__(self, aabb: AABB, primitives: list[Primitive3]):
        self.bounding_box = aabb
        # TODO: maybe use indices into the primitive array in case the primitives get copied while beeing sliced
        self.primitives = primitives

    def split(self, split_axis: Axis, axis_pos: float):
        """ Splits a node by the given split axis and its position into two nodes. """
        left_primitives = []
        right_primitives = []
        right_new_minimum = sys.float_info.max
        left_new_maximum = sys.float_info.min

        for i in range(len(self.primitives)):
            # the maximum extend of the primitive in regard to the current split
            max_primitive = max(self.primitives[i][0][split_axis], 
                                self.primitives[i][1][split_axis], 
                                self.primitives[i][2][split_axis])
            
            if max_primitive < split_axis:
                left_primitives.append(self.primitives[i])
            
            # the minimum extend of the primitive in regard to the current split
            min_primitive = min(self.primitives[i][0][split_axis], 
                                self.primitives[i][1][split_axis], 
                                self.primitives[i][2][split_axis])
            
            if min_primitive >= split_axis:
                right_primitives.append(self.primitives[i])

            # add primitives which crosses the split axis to the left if the part left of the axis 
            # is bigger than the right part, to the right otherwise
            if axis_pos - min_primitive > max_primitive - axis_pos:
                left_primitives.append(self.primitives[i])
            else:
                right_primitives.append(self.primitives[i])

        left_aabb = get_AABB_from_primitives(left_primitives)
        right_aabb = get_AABB_from_primitives(right_primitives)
        
        return BVHNode(left_aabb, left_primitives), BVHNode(right_aabb, right_primitives)
