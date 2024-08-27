# credits: https://github.com/cgaueb/nss
from dataclasses import dataclass
from typing import Dict
import copy
import heapq
import os
import nn_AABB
import nn_loss
import nn_mesh_list
import nn_parser
import nn_types
import nss_common
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

class pointcloud_stream :
    def __init__(self, pConfig, pPointCloudRootFolder, pCSV, pBatchSize) :
        self.config = pConfig
        self.pc_rootfolder = pPointCloudRootFolder
        self.df_files = pd.read_csv(pCSV)
        self.x = len(self.df_files.index)
        self.batchSize = self.x if self.x < pBatchSize else pBatchSize
        self.ref_point_clouds = None
        self.ref_names = None

    def __load_trees(self) :
        print('Caching trees...')
        point_clouds = []
        names = []

        # iterate over all point cloud files
        for df_file in self.df_files.iterrows() :
            name: str = df_file[1]['samples']
            name = name.replace('\\', os.path.sep)
            # loads the point cloud from a .npz file and saves it in the pc numpy array of points (x,y,z)
            with np.load(os.path.join(self.pc_rootfolder, name) + '.npz', allow_pickle=True) as f :
                pc = f['a']

            s = np.random.normal(0, 0.0001, pc.shape[0] * 3) # ? no purpose
            points = pc + np.reshape(s, pc.shape)            # ? no purpose
            points = pc
            # skip point cloud if volume is too small
            if(nss_common.volume(points) < 1.e-4) :
                continue

            points = nss_common.applyNormalization(pc, nss_common.getAABBox(pc), 1.0, self.config['gamma'])
            point_clouds += [points,]
            names += [name,]
        print('Tree caching finished')

        return np.array(names), np.array(point_clouds, dtype=np.float32)

    def init_dataset(self, pDropReminder=False, pShuffle = True) :
        # names of the scenes and according point clouds of the scenes which are numpy arrays of the type np.float32
        names, point_clouds = self.__load_trees()
        self.ref_point_clouds = point_clouds
        self.ref_names = names

        # dataset is tf.data.Dataset object which holds the names and the according point clouds of the scenes
        self.dataset = tf.data.Dataset.from_tensor_slices((names, point_clouds))
        # shuffles the set
        self.dataset = self.dataset.shuffle(buffer_size=self.x, reshuffle_each_iteration=pShuffle)
        # groups the data set into batches of dize batchSize (last parameter defines if the last batch should not be used if it doesn't have len=batchSize)
        self.dataset = self.dataset.batch(batch_size=self.batchSize, drop_remainder=pDropReminder)
        
        #self.dataset = tf.data.Dataset.From_generator() can be used for data augmentation
        # important! if parallelization is used -> enforce that each thread gets its own random number generator! otherwise the same scene will be generated.

        # determines that the next batch shall be generated while the current one is processed
        # -> num_parallel_calls=tf.data.AUTOTUNE can be given as additional argument to tell tf that the prefetch should be parallized.
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.dataset
    
####################################################### EDIT ##################################################################################

class Scene:
    def __init__(self, _meshes: nn_mesh_list.Mesh3List, batch_size, primitive_cloud_size: float, rng_seed):
        self.prim_cloud_size = primitive_cloud_size
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.meshes = _meshes
        self.bounds = nn_AABB.get_AABB_from_primitives(self.meshes.primitives)
        self.primitive_cloud: nn_mesh_list.Mesh3List = nn_mesh_list.Mesh3List()

        # ugly code to avoid branching in get_next_transformed_batch()
        self.bounds_list: list[float] = [] 
        self.bounds_list.append(self.bounds.x_max)
        self.bounds_list.append(self.bounds.y_max)
        self.bounds_list.append(self.bounds.z_max)
        self.bounds_list.append(self.bounds.x_min)
        self.bounds_list.append(self.bounds.y_min)
        self.bounds_list.append(self.bounds.z_min)

        non_moveable_mesh_range = self.meshes.mesh_indices[0]
        self.meshes.mesh_indices.pop(0)

        self.batch_primitives_AABB_lists = [[[] for _ in range(len(self.meshes.mesh_indices))] \
                                             for _ in range( self.batch_size)]
        for mesh_index, mesh_interval in enumerate(self.meshes.mesh_indices):
            mesh = self.meshes.primitives[mesh_interval.low : mesh_interval.up]
            aabb = nn_AABB.get_AABB_from_primitives(mesh)
            for i in range(0,  self.batch_size):
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.x_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.y_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.z_max)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.x_min)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.y_min)
                self.batch_primitives_AABB_lists[i][mesh_index].append(aabb.z_min)
        
        # take samples
        # special case for non moveable objects
        samples_non_moveable = 24 # 6 Walls * 2 Polygons per wall * 2 both sides of wall
        if samples_non_moveable < non_moveable_mesh_range.up:
            prims_with_size = []
            for i in range(non_moveable_mesh_range.up):
                prims_with_size.append({'index': i, 'size': nn_loss.surface_area_primitive(self.meshes.primitives[i])})
            largest_prims_with_size = heapq.nlargest(samples_non_moveable, prims_with_size, key=lambda x : x['size'])
            largest_prims_indices = [i['index'] for i in largest_prims_with_size]
        else: 
            largest_prims_indices = list(range(non_moveable_mesh_range.up))
        for prim_idx in largest_prims_indices:
            self.primitive_cloud.primitives.append(self.meshes.primitives[prim_idx])
        
        # moveable objects
        remaining_sample_size: float = self.prim_cloud_size - len(self.primitive_cloud.primitives)
        remaining_primitives_size: float = float(len(self.meshes.primitives)) - non_moveable_mesh_range.up
        samples_per_prim: float = min(remaining_sample_size / remaining_primitives_size, 1)
        
        step = remaining_primitives_size / remaining_sample_size
        sample_indices = [(non_moveable_mesh_range.up + round(i * step)) for i in range(remaining_sample_size)]
        
        sample_idx = 0
        removed_meshes = 0
        for mesh_index, mesh_interval in enumerate(self.meshes.mesh_indices):
            if sample_indices[sample_idx] < mesh_interval.low or sample_indices[sample_idx] >= mesh_interval.up:
                # remove AABB of the mesh from every batch entry
                for prim_AABB_list in self.batch_primitives_AABB_lists:
                    prim_AABB_list.pop(mesh_index - removed_meshes)
                removed_meshes += 1
                continue
            prim_cloud_mesh: nn_types.Mesh3 = []
            while sample_idx < len(sample_indices) \
                and mesh_interval.low <= sample_indices[sample_idx] and sample_indices[sample_idx] < mesh_interval.up:
                prim_cloud_mesh.append(self.meshes.primitives[sample_indices[sample_idx]])
                sample_idx += 1
            self.primitive_cloud.append(prim_cloud_mesh)
        
        self.batch_primitive_clouds: list[np.ndarray] = []
        for _ in range(0,  self.batch_size):
            self.batch_primitive_clouds.append(np.array(self.primitive_cloud.primitives))

        self.backup_batch_primitives_AABB_lists = copy.deepcopy(self.batch_primitives_AABB_lists)
        self.backup_batch_primitive_clouds = copy.deepcopy(self.batch_primitive_clouds)


    def reset(self):
        self.backup_batch_primitives_AABB_lists = copy.deepcopy(self.backup_batch_primitives_AABB_lists)
        self.batch_primitive_clouds = copy.deepcopy(self.backup_batch_primitive_clouds)
        self.rng = np.random.default_rng(self.rng_seed)


    def get_test_dataset(self, dataset_size):
        backup_batch_primitives_AABB_lists = copy.deepcopy(self.batch_primitives_AABB_lists)
        backup_batch_primitive_clouds = copy.deepcopy(self.batch_primitive_clouds)

        self.rng = np.random.default_rng(52535)
        test_clouds = []
        test_clouds.append(np.array(self.primitive_cloud.primitives))

        while len(test_clouds) < dataset_size:
            test_clouds.extend(self.get_next_tranformed_batch())

        test_clouds = test_clouds[:dataset_size]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            test_clouds_dataset = np.array(test_clouds).reshape(dataset_size, self.prim_cloud_size, 9)

        self.batch_primitives_AABB_lists = backup_batch_primitives_AABB_lists
        self.batch_primitive_clouds = backup_batch_primitive_clouds
        self.rng = np.random.default_rng(self.rng_seed)

        return tf.convert_to_tensor(test_clouds_dataset, dtype=tf.float32)
    

    def get_next_tranformed_batch(self) -> list[np.ndarray]:
        """ Generates a batch of transformations of the current scene and returns it. """
        meshes_len = len(self.primitive_cloud.mesh_indices)
        quart_len = int(np.floor((meshes_len) / 4))
        # shift to access the minimum axis bounds
        min = 3

        for batch_idx, primitives in enumerate(self.batch_primitive_clouds):
            transform_size = self.rng.integers(quart_len, quart_len * 3 + 1)
            # indices of the meshes that shall be transformed
            transform_indices = self.rng.choice(meshes_len, size=transform_size, replace=False)
            for m in transform_indices:
                interval = self.primitive_cloud.mesh_indices[m]
                # choose random axis: 0=x, 1=y, 2=z
                axis = self.rng.integers(0,3)

                mesh_aabb_extend = self.batch_primitives_AABB_lists[batch_idx][m][axis] \
                    - self.batch_primitives_AABB_lists[batch_idx][m][min + axis]
                halve_extend = mesh_aabb_extend / 2
                # guaranties that the new position is in bound
                lower_bound = self.bounds_list[min + axis] + halve_extend
                upper_bound = self.bounds_list[axis] - halve_extend
                # in regard to the middle point of the aabb
                new_pos = self.rng.uniform(lower_bound, upper_bound)
                current_pos = self.batch_primitives_AABB_lists[batch_idx][m][axis] - halve_extend
                             
                translation = new_pos - current_pos

                for i in range(interval.low, interval.up):
                    primitives[i][0][axis] += translation
                    primitives[i][1][axis] += translation
                    primitives[i][2][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][axis] += translation
                self.batch_primitives_AABB_lists[batch_idx][m][min + axis] += translation
        
        return self.batch_primitive_clouds
    

@dataclass
class Scene_meshNamePair:
    scene: nn_mesh_list.Mesh3List
    name: str    

class primitive_cloud_generator:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.batch_limit = config['batch_amount']
        self.train_scene_folder = config['train_scenes_dir']
        self.test_scene_folder = config['test_scenes_dir']
        self.prim_cloud_size = config['point_cloud_size']
        self.scene_iter = 0
        self.scenes: list[Scene] = []
        self.scene_names: list[str] = []
        self.test_dataset = []
        self.coordinates_order = [0,3,6,1,4,7,2,5,8]

        print("Loading train scenes...")
        for scene_file in os.listdir(self.train_scene_folder):
            if not scene_file.endswith('.obj'):
                continue
            self.scene_names.append(scene_file[:-10])
            print("Loading ", scene_file, "...")
            scene_meshes = nn_parser.parse_obj_file_with_meshes(os.path.join(self.train_scene_folder, scene_file))
            nn_parser.scale_scene(scene_meshes.primitives, 1)

            sc = Scene(scene_meshes, self.batch_size, self.prim_cloud_size, 83242)
            self.scenes.append(sc)
            #self.test_dataset.extend(sc.get_test_dataset(config['test_sets']))
        #self.test_dataset = tf.convert_to_tensor(self.test_dataset, dtype=tf.float32)
        print("... done.")

    def reset_scenes(self):
        for scene in self.scenes:
            scene.reset()

    def get_next_batch(self):
        if self.scene_iter >= self.batch_limit:
            return tf.constant([], dtype=tf.float32)
        cur_scene: Scene = self.scenes[self.scene_iter % len(self.scenes)]
        self.scene_iter += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            transformed_scenes_batch = np.array(cur_scene.get_next_tranformed_batch()).reshape(self.batch_size, self.prim_cloud_size, 9)
        
        result = tf.convert_to_tensor(transformed_scenes_batch, dtype=tf.float32)
        result = tf.gather(result, self.coordinates_order, axis=2)
        return result
    
    def get_base_scenes_for_nn_prediction(self):
        result = []
        for scene in self.scenes:
            prim_cloud = np.array(scene.backup_batch_primitive_clouds[0]).reshape(1, self.prim_cloud_size, 9)
            prim_cloud = tf.convert_to_tensor(prim_cloud, dtype=tf.float32)
            prim_cloud = tf.gather(prim_cloud, self.coordinates_order, axis=2)
            result.append(prim_cloud)

        return result, self.scene_names
    
    def get_base_scenes_for_evalutation(self) -> Dict[str, Scene_meshNamePair]:
        result = {}

        print("Loading test scenes...")
        for scene_file in os.listdir(self.test_scene_folder):
            if not scene_file.endswith('.obj'):
                continue
            print("Loading ", scene_file, "...")
            scene_meshes = nn_parser.parse_obj_file_with_meshes(os.path.join(self.test_scene_folder, scene_file))
            nn_parser.scale_scene(scene_meshes.primitives)
            scene_name = scene_file[:-9]
            result[scene_name] = Scene_meshNamePair(scene_meshes, scene_name)
        print("... done.")

        return result
