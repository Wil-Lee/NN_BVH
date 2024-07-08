from contextlib import contextmanager
from nn_parser import *
from nn_AABB import *
from nn_data_augmentation import *
from nn_types import *
import pyvista as pv
import time

@contextmanager
def bench(name):
    print(f"\nStart {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"... {name} done in: " + str(end_time - start_time) + " seconds.")

def draw_scene_bounding_box(aabb: AABB, plotter):
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

    plotter.add_mesh(poly_data, color='red', line_width=6)
    


#path = "machine_learning/conference.obj"
#path = "machine_learning/Room.obj"
path = "machine_learning/bbedroom.pbrt"



#primitives = parse_obj_file_with_primitives(path)
#aabb = get_AABB_from_primitives(primitives)
#loader = DataLoaderFromPrimitives(1000000, 1, primitives, aabb)
with bench('Parsing'):
    #p_mesh = parse_obj_file_with_meshes(path)
    p_mesh = parse_pbrt_file_with_meshes(path)
with bench('Scaling'):
    p_mesh.primitives = scale_scene(p_mesh.primitives)
primitives = p_mesh.primitives
with bench('Building AABB'):
    aabb = get_AABB_from_primitives(primitives)
batch_size = 1
data_size = 1000000
loader = DataLoaderFromMeshes(data_size, batch_size, p_mesh, aabb)

if 1:
    plotter = pv.Plotter()
    draw_scene_bounding_box(aabb, plotter)
    points = np.array([vertex for primitive in primitives for vertex in primitive])
    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    mesh = pv.PolyData(points, faces)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    plotter.show()
    plotter.clear()


for i in range(0,20):
    plotter = pv.Plotter()
    with bench(f'Generating batch (size: {batch_size}) of transformed scenes'):
        #for i in range(300):
            primitives = loader.__getitem__(0)[0]

    points = np.array([vertex for primitive in primitives for vertex in primitive])
    draw_scene_bounding_box(aabb, plotter)
    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    
    mesh = pv.PolyData(points, faces)

    plotter.add_mesh(mesh, color='lightblue', show_edges=True)

    plotter.show()
    plotter.clear()



