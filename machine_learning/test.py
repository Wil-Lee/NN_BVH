from contextlib import contextmanager
from nn_parser import *
from nn_AABB import *
from nn_data_augmentation import *
from nn_types import *
from nn_BVH import * 
from nn_loss import *
import pyvista as pv
import time

@contextmanager
def bench(name):
    print(f"\nStart {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"... {name} done in: " + str(end_time - start_time) + " seconds.")

def draw_bounding_box(aabb: AABB, plotter, color: str='red', line_width: float=6):
    points = np.array([
        [aabb.x_min, aabb.y_min, aabb.z_min],
        [aabb.x_min, aabb.y_min, aabb.z_max],
        [aabb.x_min, aabb.y_max, aabb.z_min],
        [aabb.x_min, aabb.y_max, aabb.z_max],
        [aabb.x_max, aabb.y_min, aabb.z_min],
        [aabb.x_max, aabb.y_min, aabb.z_max],
        [aabb.x_max, aabb.y_max, aabb.z_min],
        [aabb.x_max, aabb.y_max, aabb.z_max]
    ])
    edges = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5],
        [4, 6], [5, 7], [6, 7]
    ]

    line_data = []
    for line in edges:
        line_data.append(2)
        line_data.extend(line)
    
    poly_data = pv.PolyData()
    poly_data.points = points
    poly_data.lines = np.array(line_data)

    plotter.add_mesh(poly_data, color=color, line_width=line_width)

def test_BVH():
    l_prim0: list[Primitive3] = [[0.9, 0.5, 0],[2.5, 0.5, 2],[2.5, 0.5, 0]]
    r_prim0: list[Primitive3] = [[1, 1, 0.1],[1, 2, 0.1],[2, 2, 0.1]]
    r_prim1: list[Primitive3] = [[0, 3, 0.05],[1.5, 3.5, 0.05],[0, 4, 0.05]]

    prims = []
    prims.append(l_prim0)
    prims.append(r_prim0)
    prims.append(r_prim1)
    head_aabb = get_AABB_from_primitives(prims)
    head = BVHNode(head_aabb, prims)
    head.split(Axis.y, 0.75)
    left = head.left_child
    right = head.right_child

    if (len(left.primitives) == 1 and left.is_leaf and 
        left.primitives[0] == l_prim0 and
        right.primitives[0] == r_prim0 and
        right.primitives[1] == r_prim1):
        print("node split - success")
    else:
        print("node split - failure")

    inner_nodes, leaf_nodes = head.to_list()
    if inner_nodes[0] == head and leaf_nodes[0] == head.left_child and leaf_nodes[1] == head.right_child:
        print("to list - success")
    else:
        print("to list - failure")
    
    left.aabb.y_max = 2.1
    left.aabb.y_min = 0.1

    ext_prims_lay_inside_node = get_external_primitives_laying_inside_node(left)

    if len(ext_prims_lay_inside_node) == 1 and ext_prims_lay_inside_node[0] == r_prim0:
        print("Getting external primitives - success")
    else:
        print("Getting external primitives - failure")
    
#test_BVH()



#path = "machine_learning/conference.obj"
#path = "machine_learning/Room.obj"
#path = "machine_learning/bedroomNoFloor.obj"

if 1:
    path = "machine_learning/bedroom_LowPoly.obj"
    with bench('Parsing'):
        p_mesh = parse_obj_file_with_meshes(path)
    # remove floor from moveable meshes
    p_mesh.mesh_indices.pop(0)
else:
    with bench('Parsing'):
        path = "machine_learning/bbedroom.pbrt"
        p_mesh = parse_pbrt_file_with_meshes(path)

#primitives = parse_obj_file_with_primitives(path)
#aabb = get_AABB_from_primitives(primitives)
#loader = DataLoaderFromPrimitives(1000000, 1, primitives, aabb)

with bench('Scaling'):
    p_mesh.primitives = scale_scene(p_mesh.primitives)
primitives = p_mesh.primitives

with bench('Calculating surface area'):

    print(f"    Surface area: {surface_area(p_mesh.primitives)}")

with bench('Building AABB'):
    ab = get_AABB_from_primitives(primitives)
batch_size = 1
data_size = 1000000
loader = DataLoaderFromMeshes(data_size, batch_size, p_mesh, ab)

if 0:
    plotter = pv.Plotter()
    draw_bounding_box(ab, plotter)
    points = np.array([vertex for primitive in primitives for vertex in primitive])
    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    mesh = pv.PolyData(points, faces)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    plotter.show()
    plotter.clear()

if 0:
    for i in range(0,10):
        plotter = pv.Plotter()
        with bench(f'Generating batch (size: {batch_size}) of transformed scenes'):
            for i in range(1):
                primitives = loader.__getitem__(0)[0]

        points = np.array([vertex for primitive in primitives for vertex in primitive])
        draw_bounding_box(ab, plotter)
        n_faces = len(primitives)
        faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
        
        mesh = pv.PolyData(points, faces)

        plotter.add_mesh(mesh, color='lightblue', show_edges=True)

        plotter.show()
        plotter.clear()



nodes_to_split: list[BVHNode] = []

head_node = BVHNode(ab, primitives)
nodes_to_split.append(head_node)

points = np.array([vertex for primitive in primitives for vertex in primitive])
n_faces = len(primitives)
faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
mesh = pv.PolyData(points, faces)


level = 0
show_current_split_level = 0
while len(nodes_to_split) != 0:
    
    new_nodes = []
    aabbs_to_draw = []
    
    if show_current_split_level:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)

    print(f"\nStart Splitting nodes on level {level}...")
    total_time = 0
    for i, n in enumerate(nodes_to_split):

        if n.is_leaf:
            continue

        max_x_extent = (n.aabb.x_max - n.aabb.x_min)
        max_y_extent = (n.aabb.y_max - n.aabb.y_min)
        max_z_extent = (n.aabb.z_max - n.aabb.z_min)
                        
        # split the axis with the biggest extent in half
        if max_x_extent >= max_y_extent and max_x_extent >= max_z_extent:
            split_axis_pos = n.aabb.x_min + (max_x_extent / 2)
            start_time = time.time()
            n.split(Axis.x, split_axis_pos)
            end_time = time.time()
        elif max_y_extent >= max_z_extent:
            split_axis_pos = n.aabb.y_min + (max_y_extent / 2)
            start_time = time.time()
            n.split(Axis.y, split_axis_pos)
            end_time = time.time()
        else:
            split_axis_pos = n.aabb.z_min + (max_z_extent / 2) 
            start_time = time.time()
            n.split(Axis.z, split_axis_pos)
            end_time = time.time()
        
        total_time += end_time - start_time
        
        if len(n.left_child.primitives) == 0 or len(n.right_child.primitives) == 0:
            n.left_child = None
            n.right_child = None
            n.is_leaf = True
        else: 
            if __debug__:
                n.left_child.layer = n.layer + 1
                n.right_child.layer = n.layer + 1
            new_nodes.append(n.left_child)
            new_nodes.append(n.right_child)

        random_color = [np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]
        aabbs_to_draw.append((n.aabb, np.copy(random_color)))

    print(f"... Splitting nodes on level {level} done in: " + str(total_time) + " seconds.")

    if show_current_split_level:
        for ab in reversed(aabbs_to_draw):
            draw_bounding_box(ab[0], plotter, color=ab[1])
        plotter.show()
        plotter.close()
        plotter.clear()
        
    nodes_to_split = new_nodes
    level += 1

with bench('Determine external primitives laying inside node'):
    external = get_external_primitives_laying_inside_node(head_node.left_child.right_child)
with bench('Converting BVH to list'):
    inner_nodes, leaf_nodes = head_node.to_list()
if 0:
    with bench('Calculating EPO'):
        epo = EPO(head_node)
        print(f'{epo}')
with bench('Calculating SAH'):
    sah = SAH(head_node)
    print(f'{sah}')

isdf = 4
    
    