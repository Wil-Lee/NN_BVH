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
