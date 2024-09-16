from nn_types import *

class Mesh3List:
    """
    This class is a list specificly for meshes with polygons with 3 vertices.

    Attributes:
        primitives: A list of all primitives of the meshes.
        mesh_indices: A list that holds index pairs for the primitive list. One pair represents a mesh.
    """
    def __init__(self):
        self.primitives: list[Primitive3] = []
        self.mesh_indices: list[_Interval] = []
    
    def append(self, mesh: Mesh3):
        self.mesh_indices.append(_Interval(len(self.primitives), len(self.primitives) + len(mesh)))
        self.primitives.extend(mesh)

class _Interval:
    def __init__(self, low: int, up: int):
        self.low = low
        self.up = up