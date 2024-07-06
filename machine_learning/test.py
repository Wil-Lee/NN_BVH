from nn_parser import *
from nn_AABB import *
from nn_data_augmentation import *
from nn_types import *
import pyvista as pv



path = "machine_learning/conference.obj"

primitives = parse_obj_file_with_primitives(path)
aabb = get_AABB_from_primitives(primitives)
loader = DataLoaderFromPrimitives(1000000, 1, primitives, aabb)

p_mesh = parse_obj_file_with_meshes(path)
primitives = p_mesh.primitives
aabb = get_AABB_from_primitives(primitives)
loader = DataLoaderFromMeshes(1000000, 1, p_mesh, aabb)

#loader.__getitem__(0)
#test = 1


plotter = pv.Plotter()
points = np.array([vertex for primitive in primitives for vertex in primitive])
n_faces = len(primitives)
faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
mesh = pv.PolyData(points, faces)
plotter.add_mesh(mesh, color='lightblue', show_edges=True)
plotter.show()
plotter.clear()



for i in range(0,100):
    plotter = pv.Plotter()
    primitives = loader.__getitem__(0)[0]

    points = np.array([vertex for primitive in primitives for vertex in primitive])

    n_faces = len(primitives)
    faces = np.hstack([[3, 3*i, 3*i+1, 3*i+2] for i in range(n_faces)])
    
    mesh = pv.PolyData(points, faces)

    plotter.add_mesh(mesh, color='lightblue', show_edges=True)

    plotter.show()
    plotter.clear()



###morgen einen transformer schreiben, der die Szene in ein gescheites boundary parst. Nach dem einlesen von Parser().
#und noch ein paar prints einfügen und zeitmessen wie lange die einzelnen Abschnitte brauchen.
#morgen mal den stand pushen!
