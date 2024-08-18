from dataclasses import dataclass
from nn_AABB import *
from nn_types import *
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
        
        Returns true if node is not a leaf, otherwise false.
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
            if axis_pos - min_primitive >= max_primitive - axis_pos:
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

        if __debug__:
            self.left_child.layer = self.layer + 1
            self.right_child.layer = self.layer + 1

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
    """ Returns all split offsets which lead to different BVH splits. """
    def center(prim: Primitive3, axis: int) -> float:
        min_val = np.min(prim[:, axis])
        max_val = np.max(prim[:, axis])
        return (min_val + max_val) / 2
    
    def next_float(value: float) -> float:
        return np.nextafter(value, np.inf)
    
    axis = _axis.value
    center_points = np.unique([center(prim, axis) for prim in prims])
    center_points = [next_float(point) for point in center_points]
    return sorted(center_points)[:-1]

@dataclass
class BestSplit:
    cost: float = None
    offset: float = None
    axis: Axis = None
    left_overlapping: list[Primitive3] = None
    right_overlapping: list[Primitive3] = None
    
@dataclass
class NodeData:
    node: BVHNode
    overlapping_prims: list[Primitive3]
    prims_surface: float = None

import nn_loss     
def build_greedy_SAH_EPO_tree_single_thread(parent_node: BVHNode, alpha: float, levels: int):
    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=parent_node, overlapping_prims=[], prims_surface=nn_loss.surface_area(parent_node.primitives)))

    for level in range(levels):
        for node_data in nodes_hierarchy[level]:
            best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
            if not __debug__ or level > 0:
                for axis in Axis:
                    split_offsets = get_all_split_offsets(node_data.node.primitives, axis)

                    for o_idx, offset in enumerate(split_offsets):
                        epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(\
                            node_data.node, axis, offset, node_data.overlapping_prims, node_data.prims_surface)
                        sah = nn_loss.SAH_single_node(node_data.node, axis, offset)
                        cost = (1-alpha) * sah + alpha * epo

                        if (cost < best_split.cost):
                            best_split.cost = cost
                            best_split.offset = offset
                            best_split.axis = axis
                            best_split.left_overlapping = left_overlapping
                            best_split.right_overlapping = right_overlapping
            elif __debug__:
                # debug only
                best_split = BestSplit()
                best_split.axis = Axis.z
                best_split.offset = 0.04701263013749959
                node_data.node.split(best_split.axis, best_split.offset)
                l_overlapping_prims = nn_loss.get_prims_laying_inside_node(node_data.node.left_child.aabb, [])
                l_overlapping_prims.extend(nn_loss.get_prims_laying_inside_node(node_data.node.left_child.aabb, node_data.node.right_child.primitives))
                r_overlapping_prims = nn_loss.get_prims_laying_inside_node(node_data.node.right_child.aabb, [])
                r_overlapping_prims.extend(nn_loss.get_prims_laying_inside_node(node_data.node.right_child.aabb, node_data.node.left_child.primitives))
                best_split.left_overlapping = l_overlapping_prims
                best_split.right_overlapping = r_overlapping_prims
            node_data.node.split(best_split.axis, best_split.offset)
            nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, best_split.left_overlapping))
            nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, best_split.right_overlapping))
            if (len(best_split.left_overlapping) == 0):
                nodes_hierarchy[level + 1][len(nodes_hierarchy[level + 1]) - 2].prims_surface = nn_loss.surface_area(node_data.node.left_child.primitives)
            if (len(best_split.right_overlapping) == 0):
                nodes_hierarchy[level + 1][len(nodes_hierarchy[level + 1]) - 1].prims_surface = nn_loss.surface_area(node_data.node.right_child.primitives)

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

def compute_cost(task: Tuple[Axis, float, BVHNode, list[Primitive3], float, float]) \
        -> Tuple[float, Axis, float, List[Primitive3], List[Primitive3]]:
        axis, offset, node, overlapping_prims, prims_surface, alpha = task

        epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(
            node, axis, offset, overlapping_prims, prims_surface)
        sah = nn_loss.SAH_single_node(node, axis, offset)
        cost = (1-alpha) * sah + alpha * epo
        return cost, axis, offset, left_overlapping, right_overlapping


def build_greedy_SAH_EPO_tree_multi_thread(parent_node: BVHNode, alpha: float, levels: int):
    
    def parallel(node_data: NodeData):
        best_split = BestSplit(cost=sys.float_info.max, offset=-0.5)
        
        with ProcessPoolExecutor() as executor:
            futures = []
            tasks = []
            for axis in Axis:
                split_offsets = get_all_split_offsets(node_data.node.primitives, axis)
                for offset in split_offsets:
                    # shallow copy intended
                    tasks.append((axis, offset, copy.copy(node_data.node), node_data.overlapping_prims, node_data.prims_surface, alpha))

            results = executor.map(compute_cost, tasks)

            for result in results:
                cost, axis, offset, left_overlapping, right_overlapping = result
                if cost < best_split.cost:
                    best_split.cost = cost
                    best_split.offset = offset
                    best_split.axis = axis
                    best_split.left_overlapping = left_overlapping
                    best_split.right_overlapping = right_overlapping

        return best_split

    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=parent_node, overlapping_prims=[], prims_surface=nn_loss.surface_area(parent_node.primitives)))

    for level in range(levels):
        for node_data in nodes_hierarchy[level]:
            best_split = parallel(node_data)
    
            node_data.node.split(best_split.axis, best_split.offset)
            nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, best_split.left_overlapping))
            nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, best_split.right_overlapping))
            if (len(best_split.left_overlapping) == 0):
                nodes_hierarchy[level + 1][len(nodes_hierarchy[level + 1]) - 2].prims_surface = nn_loss.surface_area(node_data.node.left_child.primitives)
            if (len(best_split.right_overlapping) == 0):
                nodes_hierarchy[level + 1][len(nodes_hierarchy[level + 1]) - 1].prims_surface = nn_loss.surface_area(node_data.node.right_child.primitives)