from nn_types import *
from nn_BVH import *

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


def get_external_primitives_laying_inside_node(node: BVHNode) -> list[Primitive3]:
    """ 
    Returns a list of primitives which lay inside the nodes bounding box but are not part of the node or its children.
    Corresponds to (S\Q(n) ∩ n) of the EPO paper.
    """
    if (node.parent is None):
        return []
    parent: BVHNode = node.parent
    # is always an ancestor of the argument node (during first while loop)
    current_node: BVHNode = node
    # nodes which AABBs overlapped with argument node's AABB 
    nodes_to_check: list[BVHNode] = []

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
        
        for prim in current_node.primitives:
            # checks if at least one node of a primitive lays inside the argument node's AABB
            if (
                (node.aabb.x_max >= prim[0][0] >= node.aabb.x_min and
                 node.aabb.y_max >= prim[0][1] >= node.aabb.y_min and
                 node.aabb.z_max >= prim[0][2] >= node.aabb.z_min)
                or
                (node.aabb.x_max >= prim[1][0] >= node.aabb.x_min and
                 node.aabb.y_max >= prim[1][1] >= node.aabb.y_min and
                 node.aabb.z_max >= prim[1][2] >= node.aabb.z_min)
                or
                (node.aabb.x_max >= prim[2][0] >= node.aabb.x_min and
                 node.aabb.y_max >= prim[2][1] >= node.aabb.y_min and
                 node.aabb.z_max >= prim[2][2] >= node.aabb.z_min)
            ):
                result.append(prim)
    
    return result      

def EPO(head_node: BVHNode):
    inner_nodes, leaf_nodes = head_node.to_list()
    inner_sum: float = 0
    leaf_sum: float = 0

    for inner in inner_nodes:
        inner_sum += surface_area(get_external_primitives_laying_inside_node(inner))
    # computation cost for inner nodes
    inner_sum *= 1.2

    for leaf in leaf_nodes:
        leaf_sum += surface_area(get_external_primitives_laying_inside_node(leaf))

    total_sum = inner_sum + leaf_sum
    total_surface_area = surface_area(head_node.primitives)

    return total_sum / total_surface_area