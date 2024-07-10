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
        self.scene_bounds = aabb
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

        # ugly code to avoid branching - optimization for grid transformations
        self.scene_extends: list[float] = []
        self.scene_extends.append(width_x)
        self.scene_extends.append(width_y)
        self.scene_extends.append(width_z)
        self.tranformations = np.array([step_x, step_y, step_z])
        self.scene_bounds_list: list[float] = []
        self.scene_bounds_list.append(aabb.x_max)
        self.scene_bounds_list.append(aabb.y_max)
        self.scene_bounds_list.append(aabb.z_max)
        self.scene_bounds_list.append(aabb.x_min)
        self.scene_bounds_list.append(aabb.y_min)
        self.scene_bounds_list.append(aabb.z_min)

        self.batch_primitives: list[np.ndarray] = []
        for _ in range(0, batch_size):
            self.batch_primitives.append(np.array(meshes.primitives))

        # ugly code to avoid branching - optimization for grid transformations -> for each batch: AABBs of meshes
        self.batch_primitives_AABB_lists = [[[] for _ in range(len(meshes.mesh_indices))] for _ in range(batch_size)]

        self.batch_primitives_AABBs = [[] for _ in range(batch_size)]
        for mesh_index, mesh_interval in enumerate(meshes.mesh_indices):
            mesh = meshes.primitives[mesh_interval.low:mesh_interval.up]
            aabb = get_AABB_from_primitives(mesh)
            for i in range(0, batch_size):
                self.batch_primitives_AABBs[i].append(aabb.copy())
                # ugly code to avoid branching - optimization for grid transformations
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.x_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.y_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.z_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.x_min)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.y_min)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.z_min)


    def __len__(self):
        """
        Returns the amount batches.
        """
        return self.length
    
    
    def __getitem__(self, idx: int):
        """
        Generates a set (len = batch size) of transformations of the scene and returns it.
        """

        return self.random_scene_shift()

        meshes_len = len(self.mesh_indices)
        quart_len = int(np.floor((meshes_len) / 4))
        # shift to access the minimum axis bounds
        min = 3 

        # TODO: Add parallelization:
        for batch_idx, primitives in enumerate(self.batch_primitives):

            transform_size = np.random.randint(quart_len, quart_len * 3 + 1)
            # indices of the meshes that shall be transformed
            transform_indices = np.random.choice(meshes_len, size=transform_size, replace=False)
            for m in transform_indices:
                interval = self.mesh_indices[m]
                # choose random axis: 0=x, 1=y, 2=z
                axis = np.random.randint(0,3)
                pos_or_neg = 2 * np.random.randint(0, 2) - 1
                translation = pos_or_neg * self.tranformations[axis]

                # bound check for the shifted AABB of the mesh with index m - code is ugly to allow less branches in this loop
                if (self.batch_primitives_AABB_lists[batch_idx][m][axis] + translation > self.scene_bounds_list[axis]
                    or self.batch_primitives_AABB_lists[batch_idx][m][min + axis] + translation < self.scene_bounds_list[min + axis]):
                    # flip translation if mesh moved out of bounce and check bounds again
                    translation = -translation
                    if (self.batch_primitives_AABB_lists[batch_idx][m][axis] + translation > self.scene_bounds_list[axis]
                        and self.batch_primitives_AABB_lists[batch_idx][m][min + axis] + translation < self.scene_bounds_list[min + axis]): 
                        continue

                for i in range(interval.low, interval.up):
                    primitives[i][0][axis] += translation
                    primitives[i][1][axis] += translation
                    primitives[i][2][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][min + axis] += translation         
                
        return self.batch_primitives
    
    def random_scene_shift(self): 
        meshes_len = len(self.mesh_indices)
        quart_len = int(np.floor((meshes_len) / 4))
        # shift to access the minimum axis bounds
        min = 3

        # TODO: Add parallelization:
        for batch_idx, primitives in enumerate(self.batch_primitives):

            transform_size = np.random.randint(quart_len, quart_len * 3 + 1)
            # indices of the meshes that shall be transformed
            transform_indices = np.random.choice(meshes_len, size=transform_size, replace=False)
            for m in transform_indices:
                interval = self.mesh_indices[m]
                # choose random axis: 0=x, 1=y, 2=z
                axis = np.random.randint(0,3)
                shift_factor = np.random.rand()
                new_pos = shift_factor * self.scene_extends[axis]
                             
                translation = new_pos - self.batch_primitives_AABB_lists[batch_idx][m][axis]
                if (self.batch_primitives_AABB_lists[batch_idx][m][axis] + translation > self.scene_bounds_list[axis]
                    or self.batch_primitives_AABB_lists[batch_idx][m][min + axis] + translation < self.scene_bounds_list[min + axis]):
                    continue

                for i in range(interval.low, interval.up):
                    primitives[i][0][axis] += translation
                    primitives[i][1][axis] += translation
                    primitives[i][2][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][min + axis] += translation
        
        return self.batch_primitives