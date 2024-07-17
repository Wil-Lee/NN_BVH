from nn_AABB import *
from nn_types import *

# maybe needed for later if model is implemented
"""
class BVH:
    def __init__(self, scene_primitives: list[Primitive3], scene_aabb: AABB, max_prim_per_leaf: int=2):
        self.head: BVHNode = BVHNode(scene_aabb, scene_primitives)
        self.max_prim_per_leaf: int = max_prim_per_leaf
    
    def build_recursive(self):
        # TODO: add recursive build of bvh
        return
"""

MAX_PRIMITIVES_PER_LEAF = 2

class BVHNode:
    """
    This class represents a BVH node.

    Attributes:
        aabb: Axis aligned bounding box of this node.
        pritmitives: Primitives enclosed by the aabb.
    """
    def __init__(self, aabb: AABB, primitives: list[Primitive3]):
        self.aabb: AABB = aabb
        self.primitives: list[Primitive3] = primitives

        self.is_leaf = len(primitives) <= MAX_PRIMITIVES_PER_LEAF
        self.parent: BVHNode = None
        self.left_child: BVHNode
        self.right_child: BVHNode
        if __debug__:
            self.layer = 0

    def split(self, split_axis: Axis, axis_pos: float):
        """ 
        Splits a node into two nodes by projecting the the scene to the split_axis and splitting it at axis_pos.
        
        Returns true if node is not a leaf
                otherwise false.
        """

        if self.is_leaf:
            return False
        
        split_axis_value = split_axis.value

        left_primitives = []
        right_primitives = []

        for i in range(len(self.primitives)):
            # the maximum extend of the primitive in regard to the current split
            max_primitive = max(self.primitives[i][0][split_axis_value], 
                                self.primitives[i][1][split_axis_value], 
                                self.primitives[i][2][split_axis_value])
            
            if max_primitive < axis_pos:
                left_primitives.append(self.primitives[i])
                continue
            
            # the minimum extend of the primitive in regard to the current split
            min_primitive = min(self.primitives[i][0][split_axis_value], 
                                self.primitives[i][1][split_axis_value], 
                                self.primitives[i][2][split_axis_value])
            
            if min_primitive >= axis_pos:
                right_primitives.append(self.primitives[i])
                continue

            # add primitives which crosses the split axis to the left if the part left of the axis 
            # is bigger than the right part, to the right otherwise
            if axis_pos - min_primitive > max_primitive - axis_pos:
                left_primitives.append(self.primitives[i])
                continue
            else:
                right_primitives.append(self.primitives[i])
                continue

        left_aabb = get_AABB_from_primitives(left_primitives)
        right_aabb = get_AABB_from_primitives(right_primitives)
        self.left_child = BVHNode(left_aabb, left_primitives)
        self.left_child.parent = self
        self.right_child = BVHNode(right_aabb, right_primitives)
        self.right_child.parent = self

        return True
