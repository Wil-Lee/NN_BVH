import numpy as np
import tensorflow as tf
# ignore import error underlining from keras - it just works!
import tensorflow.keras as keras
from keras.utils import Sequence
from nn_AABB import *
from nn_mesh_list import *
from nn_types import *



seed = 83242
np.random.seed(seed)

class DataLoaderFromPrimitives(Sequence):
    """
    This class is a data loader which generates transformations inside the given aabb
    of the given scene represended by the provided primitives.
    A transformated scene will only be generated when __getitem__() is called.

    Attributes:
        data_size:  Defines how many data points (transformations of the scenes) shall be generated.
        batch_sie:  Defines the amount of data points per iteration used during the training of the neural network.
        primitives: The primitives of the scene.
        aabb:       The axis aligned bound box which encloses all transformations.     
    """
    def __init__(self, data_size: int, batch_size: int, primitives: list[Primitive3], aabb: AABB):
        self.data_size = data_size
        self.batch_size = batch_size
        self.aabb = aabb

        self.length = int(np.ceil(self.data_size / self.batch_size))
        
        # devides the aabb into equally sized cuboids
        resolution = data_size ** (1/3)
        self._width_x = (aabb.x_max - aabb.x_min)
        self._width_y = (aabb.y_max - aabb.y_min)
        self._width_z = (aabb.z_max - aabb.z_min)
        self._step_x = self._width_x / resolution
        self._step_y = self._width_y / resolution
        self._step_z = self._width_z / resolution

        self.batch_primitives : list[np.ndarray] = []
        for _ in range(0, batch_size):
            self.batch_primitives.append(np.array(primitives))

    def __len__(self):
        """
        Returns the amount batches.
        """
        return self.length
    
    
    def __getitem__(self, idx: int):
        """
        Generates a set (len = batch size) of transformations of the scene and returns it.
        """
        primitives_len = len(self.batch_primitives[0])
        quart_len = int(np.floor((primitives_len) / 4))

        # TODO: Add parallelization:
        for primitives in self.batch_primitives:
            transform_size = np.random.randint(quart_len, quart_len * 3 + 1)
            transform_indices = np.random.choice(primitives_len, size=transform_size, replace=False)

            for i in transform_indices:
                # TODO: adapt transformations later --> add rotations (and cleanup hardcoded cases)
                #       bound check doesn't work completely as intended (negative numbers) but will, 
                #       when the input scene is scaled down so that every coord lies between [0,1]
                case = np.random.randint(0,6)
                # positive x direction
                if case == 0:
                    primitives[i][0][0] = (primitives[i][0][0] + self._step_x) % self._width_x
                    primitives[i][1][0] = (primitives[i][1][0] + self._step_x) % self._width_x
                    primitives[i][2][0] = (primitives[i][2][0] + self._step_x) % self._width_x
                # negative x direction
                elif case == 1:
                    primitives[i][0][0] = (primitives[i][0][0] - self._step_x) % self._width_x
                    primitives[i][1][0] = (primitives[i][1][0] - self._step_x) % self._width_x
                    primitives[i][2][0] = (primitives[i][2][0] - self._step_x) % self._width_x
                # positive y direction
                elif case == 2:
                    primitives[i][0][1] = (primitives[i][0][1] + self._step_y) % self._width_y
                    primitives[i][1][1] = (primitives[i][1][1] + self._step_y) % self._width_y
                    primitives[i][2][1] = (primitives[i][2][1] + self._step_y) % self._width_y
                # negative y direction
                elif case == 3:
                    primitives[i][0][1] = (primitives[i][0][1] - self._step_y) % self._width_y
                    primitives[i][1][1] = (primitives[i][1][1] - self._step_y) % self._width_y
                    primitives[i][2][1] = (primitives[i][2][1] - self._step_y) % self._width_y
                # positive z direction
                elif case == 4:
                    primitives[i][0][2] = (primitives[i][0][2] + self._step_z) % self._width_z
                    primitives[i][1][2] = (primitives[i][1][2] + self._step_z) % self._width_z
                    primitives[i][2][2] = (primitives[i][2][2] + self._step_z) % self._width_z
                # negative z direction
                elif case == 5:
                    primitives[i][0][2] = (primitives[i][0][2] - self._step_z) % self._width_z
                    primitives[i][1][2] = (primitives[i][1][2] - self._step_z) % self._width_z
                    primitives[i][2][2] = (primitives[i][2][2] - self._step_z) % self._width_z
                
        return self.batch_primitives
        #batch_x = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        #batch_y = batch_x * 2
        #return batch_x, batch_y   

class DataLoaderFromMeshes(Sequence):
    """
    This class is a data loader which generates transformations inside the given aabb
    of the given scene represended by the provided meshes.

    Attributes:
        data_size:  Defines how many data points (transformations of the scenes) shall be generated.
        batch_sie:  Defines the amount of data points per iterations used during the training of the neural network.
        meshes:     The meshes of the scene.
        aabb:       The axis aligned bound box which encloses all transformations.  
        mesh_indices: TODO docs    
    """
    def __init__(self, data_size: int, batch_size: int, meshes: Mesh3List, aabb: AABB):
        self.data_size = data_size
        self.batch_size = batch_size
        self.aabb = aabb
        self.mesh_indices = meshes.mesh_indices

        self.length = int(np.ceil(self.data_size / self.batch_size))
        
        # devides the aabb of the scene into equally sized cuboids
        resolution = data_size ** (1/3)
        width_x = (aabb.x_max - aabb.x_min)
        width_y = (aabb.y_max - aabb.y_min)
        width_z = (aabb.z_max - aabb.z_min)
        step_x = width_x / resolution
        step_y = width_y / resolution
        step_z = width_z / resolution

        self.tranformations = np.array([step_x, -step_x, step_y, -step_y, step_z, -step_z])

        self.batch_primitives: list[np.ndarray] = []
        for _ in range(0, batch_size):
            self.batch_primitives.append(np.array(meshes.primitives))

        self.batch_primitives_AABBs = [[] for _ in range(batch_size)]
        for mesh_interval in meshes.mesh_indices:
            mesh = meshes.primitives[mesh_interval.low:mesh_interval.up]
            aabb = get_AABB_from_primitives(mesh)
            for i in range(0, batch_size):
                self.batch_primitives_AABBs[i].append(aabb.copy())


    def __len__(self):
        """
        Returns the amount batches.
        """
        return self.length
    
    
    def __getitem__(self, idx: int):
        """
        Generates a set (len = batch size) of transformations of the scene and returns it.
        """
        meshes_len = len(self.mesh_indices)
        quart_len = int(np.floor((meshes_len) / 4))

        # TODO: Add parallelization:
        for primitives in self.batch_primitives:
            transform_size = np.random.randint(quart_len, quart_len * 3 + 1)
            # indices of the meshes that shall be transformed
            transform_indices = np.random.choice(meshes_len, size=transform_size, replace=False)

            for m in transform_indices:
                interval = self.mesh_indices[m]
                case = np.random.randint(0,6)
                # TODO: adapt transformations later --> add rotations   
                # x direction
                if case < 2:
                    transformation = self.tranformations[case]
                    for i in range(interval.low, interval.up):
                        primitives[i][0][0] += transformation
                        primitives[i][1][0] += transformation
                        primitives[i][2][0] += transformation
        
                # y direction
                elif case < 4:
                    transformation = self.tranformations[case]
                    for i in range(interval.low, interval.up):
                        primitives[i][0][1] += transformation
                        primitives[i][1][1] += transformation
                        primitives[i][2][1] += transformation

                # z direction
                else:
                    transformation = self.tranformations[case]
                    for i in range(interval.low, interval.up):
                        primitives[i][0][2] += transformation
                        primitives[i][1][2] += transformation
                        primitives[i][2][2] += transformation
                
        return self.batch_primitives
    
    def __is_in_bounds__(self, shift: float, axis: str, mesh_aabb: AABB):
        if axis == 'x':
            if shift >= 0:
                return mesh_aabb.x_max + shift <= self.aabb.x_max
            else:
                return mesh_aabb.x_min - shift >= self.aabb.x_min
        elif axis == 'y':
            if shift >= 0:
                return mesh_aabb.y_max + shift <= self.aabb.y_max
            else:
                return mesh_aabb.y_min - shift >= self.aabb.y_min
        elif axis == 'z':
            if shift >= 0:
                return mesh_aabb.z_max + shift <= self.aabb.z_max
            else:
                return mesh_aabb.z_min - shift >= self.aabb.z_min
        
    