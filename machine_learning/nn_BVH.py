from dataclasses import dataclass
from nn_AABB import *
from nn_types import *
import nn_loss
import sys

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

    def to_list(self):
        """ Converts the BVH to a list of its nodes separated in inner and leafs. """
        inner_nodes: list[BVHNode] = []
        leaf_nodes: list[BVHNode] = []
        if self.is_leaf:
            return (inner_nodes.append(self))

        nodes_to_check: list[BVHNode] = []
        nodes_to_check.append(self)

        while len(nodes_to_check):
            current_node = nodes_to_check.pop()
            while not current_node.is_leaf:
                inner_nodes.append(current_node)
                nodes_to_check.append(current_node.right_child)
                current_node = current_node.left_child
            leaf_nodes.append(current_node)

        return inner_nodes, leaf_nodes
    
def get_all_split_offsets(prims: list[Primitive3], _axis: Axis):
    axis = _axis.value
    sorted_prims = sorted(prims, key=lambda prim: np.max(prim[:, axis]))
    return ([np.max(primitive[:, axis]) for primitive in sorted_prims])[:-1]

        
def build_greedy_SAH_EPO_tree(parent_node: BVHNode, alpha: float, levels: int):
    @dataclass
    class BestSplit:
        cost: float
        offset: float
        axis: Axis
        left_overlapping: list[Primitive3]
        right_overlapping: list[Primitive3]
    
    @dataclass
    class NodeOverlapPair:
        node: BVHNode
        overlapping_prims: list[Primitive3]

    nodes_hierarchy: list[list[NodeOverlapPair]] = [[] for _ in levels + 1]
    nodes_hierarchy[0].append(NodeOverlapPair(node=parent_node, overlapping_prims=[]))

    for level in (0, levels):
        for node_ol_pair in nodes_hierarchy[level]:
            best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
            for axis in Axis:
                split_offsets = get_all_split_offsets(node_ol_pair.node.primitives, axis)

                for offset in split_offsets:
                    epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(\
                        node_ol_pair.node, axis, offset, node_ol_pair.overlapping_prims)
                    sah = nn_loss.SAH_single_node(node_ol_pair.node, axis, offset)
                    cost = (1-alpha) * sah + alpha * epo
                    
                    if (cost < best_split.cost):
                        best_split.cost = cost
                        best_split.offset = offset
                        best_split.axis = axis
                        best_split.left_overlapping = left_overlapping
                        best_split.right_overlapping = right_overlapping
            
            node_ol_pair.node.split(best_split.axis, best_split.offset)
            nodes_hierarchy[level + 1].append(NodeOverlapPair(node_ol_pair.node.left_child, best_split.left_overlapping))
            nodes_hierarchy[level + 1].append(NodeOverlapPair(node_ol_pair.node.right_child, best_split.right_overlapping))
