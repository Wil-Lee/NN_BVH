# credits: https://github.com/cgaueb/nss
import os
import nss_common
import numpy as np
import pandas as pd
import tensorflow as tf

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
            name = df_file[1]['samples']

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