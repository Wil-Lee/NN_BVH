from dataclasses import dataclass
import math
from nn_AABB import *
from nn_types import *
import os
import sys


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

        self.is_leaf = False #len(primitives) <= MAX_PRIMITIVES_PER_LEAF
        self.parent: BVHNode = None
        self.left_child: BVHNode
        self.right_child: BVHNode
        self.split_dimension: Axis
        self.split_offset: float
        if __debug__:
            self.layer = 0

    def split(self, split_axis: Axis, axis_pos: float):
        """ 
        Splits a node into two nodes by projecting the the scene to the split_axis and splitting it at axis_pos.
        
        Returns true if node is not a leaf, otherwise false.
        """
        tight_bounds = True
        
        split_axis_value = split_axis.value

        left_primitives = []
        right_primitives = []

        for i in range(len(self.primitives)):
            # the maximum extent of the primitive in regard to the current split
            max_primitive = max(self.primitives[i][0][split_axis_value], 
                                self.primitives[i][1][split_axis_value], 
                                self.primitives[i][2][split_axis_value])
            
            if max_primitive <= axis_pos:
                left_primitives.append(self.primitives[i])
                continue
            
            # the minimum extent of the primitive in regard to the current split
            min_primitive = min(self.primitives[i][0][split_axis_value], 
                                self.primitives[i][1][split_axis_value], 
                                self.primitives[i][2][split_axis_value])
            
            if min_primitive > axis_pos:
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
        
        if tight_bounds:
            left_aabb = get_AABB_from_primitives(left_primitives)
            right_aabb = get_AABB_from_primitives(right_primitives)
        else:
            left_aabb = self.aabb.copy()
            right_aabb = self.aabb.copy()
            left_max = self.aabb.get_min(split_axis)
            right_min = self.aabb.get_max(split_axis)
            for prim in left_primitives:
                max_prim = max(prim[0][split_axis_value], 
                               prim[1][split_axis_value], 
                               prim[2][split_axis_value])
                if  max_prim > left_max:
                    left_max = max_prim
            for prim in right_primitives:
                min_prim = min(prim[0][split_axis_value], 
                               prim[1][split_axis_value], 
                               prim[2][split_axis_value])
                if  min_prim < right_min:
                    right_min = min_prim
            left_aabb.set_max(split_axis, left_max)
            right_aabb.set_min(split_axis, right_min)

        self.left_child = BVHNode(left_aabb, left_primitives)
        self.left_child.parent = self
        self.right_child = BVHNode(right_aabb, right_primitives)
        self.right_child.parent = self
        self.split_dimension = split_axis
        self.split_offset = axis_pos

        if __debug__:
            self.left_child.layer = self.layer + 1
            self.right_child.layer = self.layer + 1


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
    

    def print_tree(self):
        self.__print__(self)
    

    def __print__(self, node, node_index=0, parent_index=None, indent=""):
        if parent_index is None:
            parent_str = "Root"
        else:
            parent_str = f"Parent: {parent_index}"
        
        line_prefix = f"{indent}Node: {node_index}, {parent_str}"

        if node.is_leaf:
            print(f"{line_prefix} -> Leaf Node, Amount of primitives: {len(node.primitives)}")
        else:
            print(f"{line_prefix} -> {node.split_dimension}, Offset: {node.split_offset}")
            
            if node.left_child:
                self.__print__(node.left_child, node_index * 2 + 1, node_index, indent + "     ")
            if node.right_child:
                self.__print__(node.right_child, node_index * 2 + 2, node_index, indent + "     ")
    

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

from contextlib import contextmanager
import time
@contextmanager
def bench(name):
    print(f"\nStart {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"... {name} done in: " + str(end_time - start_time) + " seconds.")

import nn_loss
import tensorflow as tf

def build_greedy_SAH_tree_tf(root_node: BVHNode, alpha: float, levels: int, batch_size_gpu: int = 8, print_progress=False):
    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=root_node, overlapping_prims=[]))

    for level in range(levels):
        for node_index, node_data in enumerate(nodes_hierarchy[level]):
            if node_data.node.is_leaf:
                continue

            best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
            
            parent_prims = tf.constant(node_data.node.primitives, dtype=tf.float32)
            parent_prims = tf.expand_dims(parent_prims, axis=0)
            parent_prims = tf.tile(parent_prims, multiples=[batch_size_gpu, 1,1,1])
            for axis in Axis:
                
                p_aabb = node_data.node.aabb
                p_x_extent = p_aabb.x_max - p_aabb.x_min
                p_y_extent = p_aabb.y_max - p_aabb.y_min
                p_z_extent = p_aabb.z_max - p_aabb.z_min
                p_surface = tf.cast(2.0 * (p_x_extent * p_y_extent + p_x_extent * p_z_extent + p_y_extent * p_z_extent), tf.float32)
                
                parent_prims_axis_points = tf.stack([
                    parent_prims[..., 0, axis.value],
                    parent_prims[..., 1, axis.value],
                    parent_prims[..., 2, axis.value]
                ], axis=-1)

                parent_prims_mins = tf.reduce_min(parent_prims_axis_points ,axis=2, keepdims=True)
                parent_prims_maxs = tf.reduce_max(parent_prims_axis_points ,axis=2, keepdims=True)
                parent_prims_mids = parent_prims_mins + ((parent_prims_maxs - parent_prims_mins) * 0.5)

                split_offsets = tf.unique(tf.squeeze(parent_prims_mids[1]))
                split_offsets = split_offsets.y
                split_offsets = tf.sort(split_offsets)
                split_offsets = split_offsets.numpy()

                batch_offsets: np.ndarray = []
                for start in range(0, len(split_offsets), batch_size_gpu):
                    end = min(start + batch_size_gpu, len(split_offsets))
                    batch_offsets.append(np.array(split_offsets[start:end]))
                
                last_len = len(batch_offsets[len(batch_offsets) - 1])
                if last_len < batch_size_gpu:
                    fill = np.array([p_aabb.get_max(axis) for i in range(batch_size_gpu - last_len)])
                    batch_offsets[len(batch_offsets) - 1] = np.concatenate((batch_offsets[len(batch_offsets) - 1], fill), axis=None)

                for i, offset in enumerate(batch_offsets):
                    cost, cost_index = nn_loss.SAH_single_node_tf(parent_prims, parent_prims_mids, axis.value,\
                                                          tf.expand_dims(tf.cast(offset, tf.float32), axis=-1), p_surface)
                    if print_progress:
                        print(f"Level: {level}   Node: {node_index}   Axis: {axis}   Batch: {i + 1}/{len(batch_offsets)}")
                    if (cost < best_split.cost):
                        best_split.cost = cost.numpy()
                        best_split.offset = offset[cost_index]
                        best_split.axis = axis
            node_data.node.split(best_split.axis, best_split.offset)
            if len(node_data.node.left_child.primitives) > 0 or len(node_data.node.right_child.primitives) > 0: 
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, []))
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, []))
            else:
                node_data.node.is_leaf = True
                node_data.node.left_child=None
                node_data.node.right_child=None
        print("Level {} splitted.".format(level))
    
    for node_data in nodes_hierarchy[levels]:
        node_data.node.is_leaf = True


# not working because of removal of leaf check in BVHNode.split()
def build_greedy_SAH_EPO_tree_single_thread(root_node: BVHNode, alpha: float, levels: int, use_epo: bool=False):
    nodes_hierarchy: list[list[NodeData]] = [[] for _ in range(levels + 1)]
    nodes_hierarchy[0].append(NodeData(node=root_node, overlapping_prims=[]))
    
    if use_epo:
        root_surface = nn_loss.surface_area(root_node.primitives)
        for level in range(levels):
            for node_data in nodes_hierarchy[level]:
                best_split = BestSplit(cost=sys.float_info.max, offset= -0.5)
                for axis in Axis:
                    split_offsets = get_all_split_offsets(node_data.node.primitives, axis)
                    max_epo = 0
                    for o_idx, offset in enumerate(split_offsets):
                        sah = nn_loss.SAH_single_node(node_data.node, axis, offset)
                        epo, left_overlapping, right_overlapping = nn_loss.EPO_single_node(\
                            node_data.node, axis, offset, node_data.overlapping_prims, root_surface)
                        if (epo > max_epo): max_epo = epo
                        cost = (1-alpha) * sah + alpha * epo
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
    
    for node_data in nodes_hierarchy[levels]:
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
            
            cost = (1-alpha) * sah + alpha * epo
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

# not working because of removal of leaf check in BVHNode.split()
def build_greedy_SAH_EPO_tree_multi_thread(root_node: BVHNode, alpha: float, levels: int, max_workes:int = os.cpu_count(), use_epo: bool=False):

    print("Calulating root node's prims surface...")
    root_surface = nn_loss.surface_area(root_node.primitives)
    print("...done!")
    
    def parallel(node_data: NodeData, chunk_size: int):
        best_split = BestSplit(cost=sys.float_info.max, offset=-0.5)
        
        with ProcessPoolExecutor(max_workers=max_workes) as executor:
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
            chunk_size = max(4 if use_epo else 32, math.ceil(len(node_data.node.primitives) / os.cpu_count()))
            best_split = parallel(node_data, chunk_size)
    
            if node_data.node.split(best_split.axis, best_split.offset): 
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.left_child, best_split.left_overlapping))
                nodes_hierarchy[level + 1].append(NodeData(node_data.node.right_child, best_split.right_overlapping))
            else:
                node_data.node.is_leaf = True
        print("Level {} splitted.".format(level))
    
    for node_data in nodes_hierarchy[levels]:
        node_data.node.is_leaf = True