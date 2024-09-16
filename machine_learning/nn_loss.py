from nn_types import *
import nn_BVH
import tensorflow as tf

def surface_area(primitives: list[Primitive3]):
    """ Returns the total area of all primitives given in the argument list. """
    surface_area: float = 0

    for prim in primitives:
        AB = prim[1] - prim[0]
        AC = prim[2] - prim[0]
        u = np.cross(AB, AC)
        norm_u = np.linalg.norm(u)
        surface_area += norm_u
    
    surface_area *= 0.5
    
    return surface_area

def surface_area_primitive(prim: Primitive3):
    """ Returns the area of the primitive. """

    AB = prim[1] - prim[0]
    AC = prim[2] - prim[0]
    u = np.cross(AB, AC)
    surface_area: float = np.linalg.norm(u)
    surface_area *= 0.5
    
    return surface_area

def get_prims_laying_inside_node(aabb: nn_BVH.AABB, prims: list[Primitive3]):
    result = []
    for prim in prims:
        # checks if at least one node of a primitive lays inside the argument node's AABB
        if (
            (aabb.x_max >= prim[0][0] >= aabb.x_min and
             aabb.y_max >= prim[0][1] >= aabb.y_min and
             aabb.z_max >= prim[0][2] >= aabb.z_min)
            or
            (aabb.x_max >= prim[1][0] >= aabb.x_min and
             aabb.y_max >= prim[1][1] >= aabb.y_min and
             aabb.z_max >= prim[1][2] >= aabb.z_min)
            or
            (aabb.x_max >= prim[2][0] >= aabb.x_min and
             aabb.y_max >= prim[2][1] >= aabb.y_min and
             aabb.z_max >= prim[2][2] >= aabb.z_min)
        ):
            result.append(prim)
    
    return result

def get_external_primitives_laying_inside_node(node: nn_BVH.BVHNode) -> list[Primitive3]:
    """ 
    Returns a list of primitives which lay inside the nodes bounding box but are not part of the node or its children.
    Corresponds to (S\Q(n) ∩ n) of the EPO paper.
    """
    if (node.parent is None):
        return []
    parent: nn_BVH.BVHNode = node.parent
    # is always an ancestor of the argument node (during first while loop)
    current_node: nn_BVH.BVHNode = node
    # nodes which AABBs overlapped with argument node's AABB 
    nodes_to_check: list[nn_BVH.BVHNode] = []

    # iterate to the root of the BVH and collect all nodes which AABBs overlap with the argument node's AABB
    while parent is not None: # false wegmachen
        if parent.left_child == current_node:
            sibling_node = parent.right_child
        else: 
            sibling_node = parent.left_child

        # check if current_node's sibbling node's AABB overlaps with argument node's AABB
        if (node.aabb.x_min <= sibling_node.aabb.x_max and node.aabb.x_max >= sibling_node.aabb.x_min and 
            node.aabb.y_min <= sibling_node.aabb.y_max and node.aabb.y_max >= sibling_node.aabb.y_min and
            node.aabb.z_min <= sibling_node.aabb.z_max and node.aabb.z_max >= sibling_node.aabb.z_min):
            nodes_to_check.append(sibling_node)
        
        current_node = parent
        parent = parent.parent

    result: list[Primitive3] = []

    # explore overlapping nodes with dfs until leaf is hit - code is reachable
    while len(nodes_to_check):
        current_node = nodes_to_check.pop()
        while not current_node.is_leaf:
            left_child_overlapping: bool = (
                node.aabb.x_min <= current_node.left_child.aabb.x_max and node.aabb.x_max >= current_node.left_child.aabb.x_min and 
                node.aabb.y_min <= current_node.left_child.aabb.y_max and node.aabb.y_max >= current_node.left_child.aabb.y_min and
                node.aabb.z_min <= current_node.left_child.aabb.z_max and node.aabb.z_max >= current_node.left_child.aabb.z_min)
            right_child_overlapping: bool = (
                node.aabb.x_min <= current_node.right_child.aabb.x_max and node.aabb.x_max >= current_node.right_child.aabb.x_min and 
                node.aabb.y_min <= current_node.right_child.aabb.y_max and node.aabb.y_max >= current_node.right_child.aabb.y_min and
                node.aabb.z_min <= current_node.right_child.aabb.z_max and node.aabb.z_max >= current_node.right_child.aabb.z_min)
            if left_child_overlapping:
                if right_child_overlapping:
                    nodes_to_check.append(current_node.right_child)
                current_node = current_node.left_child
            elif right_child_overlapping:
                current_node = current_node.right_child
            else:
                break
        
        if not current_node.is_leaf:
            continue

        temp_result = get_prims_laying_inside_node(node.aabb, current_node.primitives)
        result.extend(temp_result)
    
    return result


C_inn = 1.2
"""Computation cost for inner nodes."""

C_tri = 1.0
"""Computation cost for leaf nodes."""

def EPO(head_node: nn_BVH.BVHNode):
    inner_nodes, leaf_nodes = head_node.to_list()
    inner_sum: float = 0
    leaf_sum: float = 0

    for inner in inner_nodes:
        inner_sum += surface_area(get_external_primitives_laying_inside_node(inner))
    inner_sum *= C_inn

    for leaf in leaf_nodes:
        leaf_sum += surface_area(get_external_primitives_laying_inside_node(leaf))
    leaf_sum *= C_tri

    total_sum = inner_sum + leaf_sum
    total_surface_area = surface_area(head_node.primitives)

    return total_sum / total_surface_area


def EPO_single_node(parent_node: nn_BVH.BVHNode, split_axis: Axis, axis_pos: float, p_overlapping_prims: list[Primitive3]=[], root_prims_surface: float=0):
    parent_node.split(split_axis, axis_pos)

    p_aabb = parent_node.aabb
    p_x_extent = p_aabb.x_max - p_aabb.x_min
    p_y_extent = p_aabb.y_max - p_aabb.y_min
    p_z_extent = p_aabb.z_max - p_aabb.z_min
    p_surface = 2.0 * (p_x_extent * p_y_extent + p_x_extent * p_z_extent + p_y_extent * p_z_extent)

    l_overlapping_prims = get_prims_laying_inside_node(parent_node.left_child.aabb, p_overlapping_prims)
    l_overlapping_prims.extend(get_prims_laying_inside_node(parent_node.left_child.aabb, parent_node.right_child.primitives))
    r_overlapping_prims = get_prims_laying_inside_node(parent_node.right_child.aabb, p_overlapping_prims)
    r_overlapping_prims.extend(get_prims_laying_inside_node(parent_node.right_child.aabb, parent_node.left_child.primitives))

    l_surface = surface_area(l_overlapping_prims)
    r_surface = surface_area(r_overlapping_prims)

    l_prim_count = len(parent_node.left_child.primitives)
    r_prim_count = len(parent_node.right_child.primitives)

    parent_node.left_child = None
    parent_node.right_child = None

    return (((l_surface / p_surface) * l_prim_count) + ((r_surface / p_surface) * r_prim_count)) * C_tri * 0.5, \
        l_overlapping_prims, r_overlapping_prims
    

def SAH(head_node: nn_BVH.BVHNode):
    inner_nodes, leaf_nodes = head_node.to_list()
    inner_sum: float = 0
    leaf_sum: float = 0

    for inner in inner_nodes:
        x_extent = inner.aabb.x_max - inner.aabb.x_min
        y_extent = inner.aabb.y_max - inner.aabb.y_min
        z_extent = inner.aabb.z_max - inner.aabb.z_min
        # adds AABB surface area
        inner_sum += 2 * (x_extent * y_extent + x_extent * z_extent + y_extent * z_extent)
    inner_sum *= C_inn

    for leaf in leaf_nodes:
        x_extent = leaf.aabb.x_max - leaf.aabb.x_min
        y_extent = leaf.aabb.y_max - leaf.aabb.y_min
        z_extent = leaf.aabb.z_max - leaf.aabb.z_min
        # adds AABB surface area * amount of primitives of the leaf node
        leaf_sum += (2 * (x_extent * y_extent + x_extent * z_extent + y_extent * z_extent)) * len(leaf.primitives)
    leaf_sum *= C_tri

    total_sum = inner_sum + leaf_sum
    x_extent = head_node.aabb.x_max - head_node.aabb.x_min
    y_extent = head_node.aabb.y_max - head_node.aabb.y_min
    z_extent = head_node.aabb.z_max - head_node.aabb.z_min
    head_aabb_surface_area = 2 * (x_extent * y_extent + x_extent * z_extent + y_extent * z_extent)

    return total_sum / head_aabb_surface_area

def SAH_single_node(parent_node: nn_BVH.BVHNode, split_axis: Axis, axis_pos: float):
    """ Returns the SAH cost of a single split. """
    parent_node.split(split_axis, axis_pos)

    p_aabb = parent_node.aabb
    p_x_extent = p_aabb.x_max - p_aabb.x_min
    p_y_extent = p_aabb.y_max - p_aabb.y_min
    p_z_extent = p_aabb.z_max - p_aabb.z_min
    p_surface = 2.0 * (p_x_extent * p_y_extent + p_x_extent * p_z_extent + p_y_extent * p_z_extent)

    l_aabb = parent_node.left_child.aabb
    l_x_extent = l_aabb.x_max - l_aabb.x_min
    l_y_extent = l_aabb.y_max - l_aabb.y_min
    l_z_extent = l_aabb.z_max - l_aabb.z_min
    l_surface = 2.0 * (l_x_extent * l_y_extent + l_x_extent * l_z_extent + l_y_extent * l_z_extent)

    r_aabb = parent_node.right_child.aabb
    r_x_extent = r_aabb.x_max - r_aabb.x_min
    r_y_extent = r_aabb.y_max - r_aabb.y_min
    r_z_extent = r_aabb.z_max - r_aabb.z_min
    r_surface = 2.0 * (r_x_extent * r_y_extent + r_x_extent * r_z_extent + r_y_extent * r_z_extent)

    l_prim_count = len(parent_node.left_child.primitives)
    r_prim_count = len(parent_node.right_child.primitives)
    
    parent_node.left_child = None
    parent_node.right_child = None

    return (((l_surface / p_surface) * l_prim_count) + ((r_surface / p_surface) * r_prim_count)) * C_tri


tf_C_tri = tf.cast(C_tri, tf.float32)

def SAH_single_node_tf(parent_prims, parent_prims_mids, split_axis, split_pos, parent_surface):
    left_child_mask = tf.cast(parent_prims_mids <= split_pos[..., tf.newaxis], tf.float32)
    right_child_mask = tf.cast(parent_prims_mids > split_pos[..., tf.newaxis], tf.float32)
    
    left_prims = tf.where(left_child_mask[..., tf.newaxis] == 1, parent_prims, tf.zeros_like(parent_prims))
    right_prims = tf.where(right_child_mask[..., tf.newaxis] == 1, parent_prims, tf.zeros_like(parent_prims))

    left_prims_count = tf.squeeze(tf.reduce_sum(left_child_mask, axis=1))
    right_prims_count = tf.squeeze(tf.reduce_sum(right_child_mask, axis=1))

    left_x_values = left_prims[..., 0]
    left_y_values = left_prims[..., 1]
    left_z_values = left_prims[..., 2]

    left_x_min = tf.reduce_min(left_x_values + right_child_mask, axis=[1,2])
    left_x_max = tf.reduce_max(left_x_values, axis=[1,2])
    left_y_min = tf.reduce_min(left_y_values + right_child_mask, axis=[1,2])
    left_y_max = tf.reduce_max(left_y_values, axis=[1,2])
    left_z_min = tf.reduce_min(left_z_values + right_child_mask, axis=[1,2])
    left_z_max = tf.reduce_max(left_z_values, axis=[1,2])

    left_x_extent = left_x_max - left_x_min
    left_y_extent = left_y_max - left_y_min
    left_z_extent = left_z_max - left_z_min
    left_surface = 2.0 * (left_x_extent * left_y_extent + left_x_extent * left_z_extent + left_y_extent * left_z_extent)

    right_x_values = right_prims[..., 0]
    right_y_values = right_prims[..., 1]
    right_z_values = right_prims[..., 2]
    
    right_x_min = tf.reduce_min(right_x_values + left_child_mask, axis=[1,2])
    right_x_max = tf.reduce_max(right_x_values, axis=[1,2])
    right_y_min = tf.reduce_min(right_y_values + left_child_mask, axis=[1,2])
    right_y_max = tf.reduce_max(right_y_values, axis=[1,2])
    right_z_min = tf.reduce_min(right_z_values + left_child_mask, axis=[1,2])
    right_z_max = tf.reduce_max(right_z_values, axis=[1,2])

    right_x_extent = right_x_max - right_x_min
    right_y_extent = right_y_max - right_y_min
    right_z_extent = right_z_max - right_z_min
    right_surface = 2.0 * (right_x_extent * right_y_extent + right_x_extent * right_z_extent + right_y_extent * right_z_extent)

    cost = (((left_surface / parent_surface) * left_prims_count) + ((right_surface / parent_surface) * right_prims_count)) * tf_C_tri

    min_cost = tf.reduce_min(cost)
    min_cost_index = tf.argmin(cost, axis=0)

    return min_cost, min_cost_index