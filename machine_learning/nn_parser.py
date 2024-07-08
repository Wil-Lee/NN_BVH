from nn_AABB import *
from nn_mesh_list import *
from nn_types import *
import numpy as np

def parse_pbrt_file_with_meshes(file_path: str):
    """ Parses the provided file into a set of meshes. """
    mesh_list = Mesh3List()
    
    mesh_idx = -1
    mesh_indices : list[list[int]] = []
    vertices_list: list[list[Vertex3]] = []

    reading_indices: bool = False
    reading_vertices: bool = False

    with open(file_path, 'r') as pbrt_file:
        for l in pbrt_file:
            line = l.strip().split()

            if not len(line):
                continue

            if reading_indices:
                if line [len(line) - 1] == "]":
                    reading_indices = False
                    for i in range(0, len(line) - 1):
                        mesh_indices[mesh_idx].append(int(line[i]))
                    continue
                for idx in line:
                    mesh_indices[mesh_idx].append(int(idx))
                continue

            if reading_vertices:
                if line [len(line) - 1] == "]":
                    reading_vertices= False
                    for i in range(0, len(line) - 1, 3):
                        vertices_list[mesh_idx].append(
                            np.array([float(line[i]), float(line[i+1]), float(line[i+2])])
                            )
                    continue
                for i in range(0, len(line), 3):
                    vertices_list[mesh_idx].append(
                        np.array([float(line[i]), float(line[i+1]), float(line[i+2])])
                        )


            if (line[0] != "\"integer" or line[1] != "indices\"") and (line[0] != "\"point3" or line[1] != "P\"" ):
                continue

            if line[0] == "\"integer":
                reading_indices = True
                mesh_indices.append([])
                mesh_idx += 1

                mesh_indices[mesh_idx].append(int(line[2].split('[')[1]))

                # special case for one liner
                length = len(line)
                if line[length - 1] == ']':
                    length -= 1
                    reading_indices = False

                for i in range(3, length):
                    mesh_indices[mesh_idx].append(int(line[i]))
                continue

            if line[0] == "\"point3":
                reading_vertices = True
                vertices_list.append([])

                vertices_list[mesh_idx].append(
                    np.array([float(line[2].split('[')[1]), float(line[3]), float(line[4])])
                    )

                # special case for one liner
                length = len(line)
                if line[length - 1] == ']':
                    length -= 1
                    reading_vertices = False

                for i in range(5, len(line), 3):
                    vertices_list[mesh_idx].append(
                        np.array([float(line[i]), float(line[i+1]), float(line[i+2])])
                        )
                continue
    
    for mesh_index in range(0, len(mesh_indices)):
        indices = mesh_indices[mesh_index]
        vertices = vertices_list[mesh_index]
        mesh: Mesh3 = []
        for i in range(0, len(indices), 3):
            mesh.append(
                np.array([vertices[indices[i]], vertices[indices[i+1]], vertices[indices[i+2]]])
                )
        mesh_list.append(mesh)

    return mesh_list

"""
def parse_obj_file_with_primitives(file_path: str):
    \""" NOT WORKING!!! Parses the provided file into a set of primitives. \"""
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
"""

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
    
