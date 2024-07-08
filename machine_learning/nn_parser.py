from nn_mesh_list import *
from nn_types import *
import numpy as np
    

def parse_obj_file_with_primitives(file_path: str):
    """ Parses the provided file into a set of primitives. """
    vertices: list[Vertex3] = []
    primitives: list[Primitive3] = []
    with open(file_path, 'r') as obj_file:

        for l in obj_file:
            # remove strip if parsing takes too long
            line = l.strip().split()
            if not len(line):
                continue

            if line[0] == 'v':
                vertices.append(np.array( [float(line[1]), float(line[2]), float(line[3])] ))
                continue 

            if line[0] == 'f':
                vertex_indices = [] # for the current face
                for i in range(1, len(line)):
                    vertex_indices.append(int(line[i].split('/')[0]) - 1)

                primitives.append(np.array([
                    vertices[vertex_indices[0]], 
                    vertices[vertex_indices[1]], 
                    vertices[vertex_indices[2]]
                    ]))
    return primitives


def parse_obj_file_with_meshes(file_path: str):
    """ Parses the provided file into a set of meshes. """
    mesh_list = Mesh3List()
    
    mesh: Mesh3 = []
    mesh_idx = -1
    vertices: list[Vertex3] = []

    with open(file_path, 'r') as obj_file:

        for l in obj_file:
            # remove strip if parsing takes too long
            line = l.strip().split()
            if not len(line):
                continue

            if line[0] == 'v':
                vertices.append(np.array( [float(line[1]), float(line[2]), float(line[3])] ))
                continue 

            if line[0] == 'f':
                vertex_indices = [] # for the current face
                for i in range(1, len(line)):
                    vertex_indices.append(int(line[i].split('/')[0]) - 1)

                mesh.append(np.array([
                    vertices[vertex_indices[0]], 
                    vertices[vertex_indices[1]], 
                    vertices[vertex_indices[2]]
                    ]))
                continue

            if line[0] == 'o':
                mesh_list.append(mesh)
                mesh_idx += 1
                mesh = []
    
    # last mesh was not added during loop
    mesh_list.append(mesh)
    # correct unwanted side effect caused by loop
    del mesh_list.mesh_indices[0]

    return mesh_list


def scale_scene(primitives: list[Primitive3], lower_limit: int = 0, upper_limit: int = 1):
    """ Scales the largest dimension of the scene down to [lower_limit, upper_limit] and all other dimension accordingly. """

    bounds = get_AABB_from_primitives(primitives)
    
    # shift scene into positive space
    if bounds.x_min < 0:
        x_shift = abs(bounds.x_min)
        for primitive3 in primitives:
            primitive3[0][0] += x_shift
            primitive3[1][0] += x_shift
            primitive3[2][0] += x_shift
    if bounds.y_min < 0:
        y_shift = abs(bounds.y_min)
        for primitive3 in primitives:
            primitive3[0][1] += y_shift
            primitive3[1][1] += y_shift
            primitive3[2][1] += y_shift
    if bounds.z_min < 0:
        z_shift = abs(bounds.z_min)
        for primitive3 in primitives:
            primitive3[0][2] += z_shift
            primitive3[1][2] += z_shift
            primitive3[2][2] += z_shift

    # scale scene
    max_extent = bounds.x_max - bounds.x_min
    y_extent = bounds.y_max - bounds.y_min
    z_extent = bounds.z_max - bounds.z_min
    if y_extent > max_extent:
        max_extent = y_extent
    if z_extent > max_extent:
        max_extent = z_extent
    
    for p in primitives:
        p[0][0] /= max_extent
        p[0][1] /= max_extent
        p[0][2] /= max_extent
        p[1][0] /= max_extent
        p[1][1] /= max_extent
        p[1][2] /= max_extent
        p[2][0] /= max_extent
        p[2][1] /= max_extent
        p[2][2] /= max_extent


    return primitives
    
