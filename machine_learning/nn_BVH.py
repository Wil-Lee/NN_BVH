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

import nn_loss     
def build_greedy_SAH_EPO_tree_single_thread(root_node: BVHNode, alpha: float, levels: int, use_epo: bool=False):
    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=root_node, overlapping_prims=[]))

    root_surface = nn_loss.surface_area(root_node.primitives)
    if use_epo:
        for level in range(levels):
            for node_data in nodes_hierarchy[level]:
                best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
                for axis in Axis:
                    split_offsets = get_all_split_offsets(node_data.node.primitives, axis)

                    for o_idx, offset in enumerate(split_offsets):
                        sah = nn_loss.SAH_single_node(node_data.node, axis, offset)
                        epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(\
                            node_data.node, axis, offset, node_data.overlapping_prims, root_surface)
                        epo_scaled = sah * epo
                        cost = (1-alpha) * sah + alpha * epo_scaled

                        if (cost < best_split.cost):
                            best_split.cost = cost
                            best_split.offset = offset
                            best_split.axis = axis
                            best_split.left_overlapping = left_overlapping
                            best_split.right_overlapping = right_overlapping
                if node_data.node.split(best_split.axis, best_split.offset): 
                    nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, best_split.left_overlapping))
                    nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, best_split.right_overlapping))
                else:
                    node_data.node.is_leaf = True
            print("Level {} splitted.".format(level))
    else:
        for level in range(levels):
            for node_data in nodes_hierarchy[level]:
                best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
                for axis in Axis:
                    split_offsets = get_all_split_offsets(node_data.node.primitives, axis)

                    for o_idx, offset in enumerate(split_offsets):    
                        cost = nn_loss.SAH_single_node(node_data.node, axis, offset)
                        if (cost < best_split.cost):
                            best_split.cost = cost
                            best_split.offset = offset
                            best_split.axis = axis
                if node_data.node.split(best_split.axis, best_split.offset): 
                    nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, []))
                    nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, []))
                else:
                    node_data.node.is_leaf = True
            print("Level {} splitted.".format(level))
    
    for node_data in nodes_hierarchy[levels - 1]:
        node_data.node.is_leaf = True

    

import copy
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

def compute_cost_with_epo(task: Tuple[Axis, float, BVHNode, list[Primitive3], float, float]) \
        -> Tuple[float, float, float, Axis, float, List[Primitive3], List[Primitive3]]:
        axis, offsets, node, overlapping_prims, root_surface, alpha = task
        best_split = BestSplit(cost=sys.float_info.max, offset=-0.5)
        best_epo = -1
        best_sah = -1
        for offset in offsets:
            sah = nn_loss.SAH_single_node(node, axis, offset)
            epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(
                node, axis, offset, overlapping_prims, root_surface)
            epo_scaled = sah * epo
            cost = (1-alpha) * sah + alpha * epo_scaled

            if cost < best_split.cost:
                best_split.cost = cost
                best_split.offset = offset
                best_split.axis = axis
                best_split.left_overlapping = left_overlapping
                best_split.right_overlapping = right_overlapping
                best_epo = epo
                best_sah = sah
        
        return best_split.cost, best_sah, best_epo, best_split.axis, best_split.offset,\
            best_split.left_overlapping, best_split.right_overlapping

def compute_cost_without_epo(task: Tuple[Axis, float, BVHNode, list[Primitive3], float, float]) \
        -> Tuple[float, float, float, Axis, float, List[Primitive3], List[Primitive3]]:
        axis, offsets, node, overlapping_prims, root_surface, alpha = task
        best_split = BestSplit(cost=sys.float_info.max, offset=-0.5)
        for offset in offsets:
            cost = nn_loss.SAH_single_node(node, axis, offset)
            if cost < best_split.cost:
                best_split.cost = cost
                best_split.offset = offset
                best_split.axis = axis
        return  best_split.cost, best_split.cost, 0, best_split.axis, best_split.offset, [], []

import os

def build_greedy_SAH_EPO_tree_multi_thread(root_node: BVHNode, alpha: float, levels: int, use_epo: bool=False):

    root_surface = nn_loss.surface_area(root_node.primitives)
    chunk_size = 16
    
    def parallel(node_data: NodeData):
        best_split = BestSplit(cost=sys.float_info.max, offset=-0.5)
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            tasks = []
            for axis in Axis:
                split_offsets = get_all_split_offsets(node_data.node.primitives, axis)
                split_offset_chunks = [split_offsets[i:i + chunk_size] for i in range(0, len(split_offsets), chunk_size)]
                for chunk in split_offset_chunks:
                    # shallow copy intended
                    tasks.append((axis, chunk, copy.copy(node_data.node), node_data.overlapping_prims, root_surface, alpha))

            if use_epo:
                results = executor.map(compute_cost_with_epo, tasks)
            else:
                results = executor.map(compute_cost_without_epo, tasks)

            for result in results:
                cost, sah, epo, axis, offset, left_overlapping, right_overlapping = result
                if cost < best_split.cost:
                    best_split.cost = cost
                    best_split.offset = offset
                    best_split.axis = axis
                    best_split.left_overlapping = left_overlapping
                    best_split.right_overlapping = right_overlapping

        return best_split

    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=root_node, overlapping_prims=[]))

    for level in range(levels):
        for node_data in nodes_hierarchy[level]:
            best_split = parallel(node_data)
    
            if node_data.node.split(best_split.axis, best_split.offset): 
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, best_split.left_overlapping))
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, best_split.right_overlapping))
            else:
                node_data.node.is_leaf = True
        print("Level {} splitted.".format(level))
    
    for node_data in nodes_hierarchy[levels - 1]:
        node_data.node.is_leaf = True