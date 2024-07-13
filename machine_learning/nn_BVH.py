from nn_AABB import *
from nn_types import *

class BVHNode:
    def __init__(self, aabb: AABB, primitives: list[Primitive3]):
        self.bounding_box = aabb
        # maybe use indices into the primitive array in case the primitives get copied while beeing sliced
        self.primitives = primitives

    def split(self, split_axis: Axis, axis_pos: float):
        # maybe it would be faster to just sort the primitives directly instead of making a copy
        sorted_primitives = sorted(self.primitives, key=self.__primitive_comperator_max__(split_axis=split_axis), reverse=False)

        lower_split_index = self.__binary_search__(sorted_primitives, split_axis, axis_pos)


        for i in range(lower_split_index, len(sorted_primitives)):
            if min(sorted_primitives[i][0][split_axis], 
                   sorted_primitives[i][1][split_axis], 
                   sorted_primitives[i][2][split_axis]):
                i = 3



        return BVHNode
    
    def __primitive_comperator_max__(primitive: Primitive3, split_axis: Axis):
        return max(primitive[0][split_axis], primitive[1][split_axis], primitive[2][split_axis])
    
    def __primitive_comperator_min__(primitive: Primitive3, split_axis: Axis):
        return min(primitive[0][split_axis], primitive[1][split_axis], primitive[2][split_axis])
    

    
    def __binary_search__(primitives: list[Primitive3], split_axis: Axis, axis_pos: float):
        """ Returns the index of the of the first primitive which crosses the split axis. """
        left = 0
        right = len(primitives) - 1
        
        while left < right:
            mid = (left + right) // 2

            mid_max_vertex = max(primitives[mid][0][split_axis], primitives[mid][1][split_axis], primitives[mid][2][split_axis]) 
            
            # maybe it would be faster to create a aabb for each primitive at the top level of the bvh to avoid these max calls 
            if ( mid_max_vertex < axis_pos
                and axis_pos <= max(primitives[mid + 1][0][split_axis], primitives[mid + 1][1][split_axis], primitives[mid + 1][2][split_axis])):
                return mid + 1
           
            elif mid_max_vertex > axis_pos:
                right = mid - 1
            
            else:
                left = mid + 1
        
        return 0
