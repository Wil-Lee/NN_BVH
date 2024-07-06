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


path = "machine_learning/conference.obj"

#primitives = parse_obj_file_with_primitives(path)
#aabb = get_AABB_from_primitives(primitives)
#loader = DataLoaderFromPrimitives(1000000, 1, primitives, aabb)
with bench('Parsing'):
    p_mesh = parse_obj_file_with_meshes(path)
primitives = p_mesh.primitives
with bench('Building AABB'):
    aabb = get_AABB_from_primitives(primitives)
batch_size = 1
loader = DataLoaderFromMeshes(1000000, batch_size, p_mesh, aabb)

if 0:
    plotter = pv.Plotter()
    points = np.array([vertex for primitive in primitives for vertex in primitive])
    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    mesh = pv.PolyData(points, faces)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    plotter.show()
    plotter.clear()


for i in range(0,1):
    plotter = pv.Plotter()
    with bench(f'Generating batch (size: {batch_size}) of transformed scenes'):
        primitives = loader.__getitem__(0)[0]

    points = np.array([vertex for primitive in primitives for vertex in primitive])

    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    
    mesh = pv.PolyData(points, faces)

    plotter.add_mesh(mesh, color='lightblue', show_edges=True)

    plotter.show()
    plotter.clear()
