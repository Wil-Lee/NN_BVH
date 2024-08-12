import tensorflow as tf
import nss_tree_common
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten

class recursive_tree_level_encoder(tf.keras.layers.Layer) :
    def __init__(self, lvl, **pConfig) :
        super(recursive_tree_level_encoder, self).__init__(name='tree_level_encoder_{0}'.format(lvl))

        self.projection_layer = self._get_linear2D(1, 'proj_layer_' + str(lvl), activ='linear', kernel_init='glorot_uniform')
        self.layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer2 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer3 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_3_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer4 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_4_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer5 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_5_' + str(lvl), activ='relu', kernel_init='he_uniform')

        self.offset_layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'regr_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer2 = self._get_linear2D(pConfig['dense_units_point_enc'] // 2, 'regr_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer3 = self._get_linear2D(1, 'regr_layer_3_' + str(lvl), activ='linear', kernel_init='glorot_uniform')

        self.lvl = lvl
        self.gamma = tf.constant([pConfig['layer_gamma']], dtype=tf.float32)
        self.beta = tf.constant([pConfig['beta']], dtype=tf.float32)
       
        self.flatten = Flatten()

    def _get_linear2D(self, filter_size, layer_name, activ, kernel_init, use_bias=False) :
        return Conv2D(filters=filter_size,
            kernel_size=(1, 1), strides=(1, 1), padding='valid',
            kernel_initializer=kernel_init,
            activation=activ,
            use_bias=use_bias,
            name=layer_name)
    
    @tf.function
    def map_from_to_opt(self, points, old_min, old_max, new_min, new_max) :
        """ Scales points to [0,1) """
        do = old_max - old_min
        dn = new_max - new_min
        a = tf.math.divide_no_nan(dn, do) # scaling factor
        # translates the points back to origin, scales them by a and translates them to new_min
        return tf.einsum('bpjk, bijk -> bpjk', (points - old_min), a) + new_min

    @tf.function
    def object_normalize(self, pc, mask) : # pc = point_cloud
        # TODO: adapt min and max to work with nodes instead of coords
        xyz_axis = tf.expand_dims(pc, axis=-1) # adds one dimension to point cloud vec (pc)
        xyz_min = tf.reduce_min(tf.abs(xyz_axis - self.beta), axis=1, keepdims=True) + self.beta
        xyz_max = tf.reduce_max(xyz_axis, axis=1, keepdims=True)
        xyz_min = tf.math.minimum(xyz_min, xyz_max)

        # TODO: same for this function
        # scales points (inside bounds) to [0,1), scales them by gamma (=1) and translates them by +(1,1,1)
        features = self.map_from_to_opt(xyz_axis,
            xyz_min, xyz_max,
            tf.zeros_like(xyz_min), tf.ones_like(xyz_min)) * self.gamma + 1

        # sets all points which are not inside the current bounding volume to 0
        features = tf.einsum('bijf, bik -> bijf', features, mask)
        return (features, tf.squeeze(xyz_min), tf.squeeze(xyz_max))

    #@tf.function
    def call(self, input) :
        point_cloud, node_bounds = input

        node_bmin = node_bounds[:, 0:3] # TODO: add min AABB
        node_bmax = node_bounds[:, 3:6] # TODO: add max AABB

        # TODO: adapt here to mask primitives
        # mask: bool vec whichs entries are 1 if the current node is inside the bound and 0 else
        node_mask = tf.stop_gradient(nss_tree_common.build_mask(point_cloud, node_bounds))

        # This code from here on implemnts Fig 1 of the paper
        #   sets all points which are not inside the current bounding volume to 0
        point_cloud = tf.stop_gradient(tf.einsum('bij, bik -> bij', point_cloud, node_mask))
        # features are scaled to [0,1) and translated by +(1,1,1)
        # min_values are probably the minimal bounding box values before the scaling (x_min, y_min, z_min). Accordingly with max_values.
        features, min_values, max_values = self.object_normalize(point_cloud, node_mask)

        # the acutal convolution is performed here:
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        
        sum_per_dim = tf.reduce_sum(features, axis=1)
        denom = tf.math.divide_no_nan(1.0, tf.reduce_sum(node_mask, axis=1, keepdims=True))
        global_descr = denom * sum_per_dim
        global_descr = global_descr[:, tf.newaxis, :, :]

        # corresponds to Conv1Ds of Fig 1 of the paper
        local_thetas = self.offset_layer1(global_descr)
        local_thetas = self.offset_layer2(local_thetas)
        local_thetas = self.offset_layer3(local_thetas)
        local_thetas = self.flatten(local_thetas)

        s = tf.stop_gradient(tf.math.divide_no_nan(max_values - min_values, node_bmax - node_bmin))
        t = tf.stop_gradient(tf.math.divide_no_nan(min_values - node_bmin, node_bmax - node_bmin))

        return local_thetas, s, t
    

class recursive_tree_level_encoder_EPO(tf.keras.layers.Layer) :
    def __init__(self, lvl, **pConfig) :
        super(recursive_tree_level_encoder_EPO, self).__init__(name='tree_level_encoder_{0}'.format(lvl))

        self.projection_layer = self._get_linear2D(1, 'proj_layer_' + str(lvl), activ='linear', kernel_init='glorot_uniform')
        self.layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer2 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer3 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_3_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer4 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_4_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer5 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_5_' + str(lvl), activ='relu', kernel_init='he_uniform')

        self.offset_layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'regr_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer2 = self._get_linear2D(pConfig['dense_units_point_enc'] // 2, 'regr_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer3 = self._get_linear2D(1, 'regr_layer_3_' + str(lvl), activ='linear', kernel_init='glorot_uniform')

        self.lvl = lvl
        self.gamma = tf.constant([pConfig['layer_gamma']], dtype=tf.float32)
        self.beta = tf.constant([pConfig['beta']], dtype=tf.float32)
       
        self.flatten = Flatten()

    def _get_linear2D(self, filter_size, layer_name, activ, kernel_init, use_bias=False) :
        return Conv2D(filters=filter_size,
            kernel_size=(1, 1), strides=(1, 1), padding='valid',
            kernel_initializer=kernel_init,
            activation=activ,
            use_bias=use_bias,
            name=layer_name)
    
    #@tf.function
    def map_from_to_opt(self, points, old_min, old_max, new_min, new_max) :
        """ Scales points to [0,1) """
        do = old_max - old_min
        dn = new_max - new_min
        a = tf.math.divide_no_nan(dn, do) # scaling factor
        a = tf.tile(a, [1,1,3,1])
        old_min = tf.tile(old_min, [1,1,3,1])
        new_min = tf.tile(new_min, [1,1,3,1])
        return tf.einsum('bpjk, bijk -> bpjk', (points - old_min), a) + new_min

    #@tf.function
    def object_normalize(self, pc, mask):
        
        xyz_axis = tf.expand_dims(pc, axis=-1)

        x_coords = tf.gather(xyz_axis, indices=[0, 3, 6], axis=2)
        y_coords = tf.gather(xyz_axis, indices=[1, 4, 7], axis=2)
        z_coords = tf.gather(xyz_axis, indices=[2, 5, 8], axis=2)
        min_x = tf.reduce_min(tf.reduce_min(tf.abs(x_coords - self.beta), axis=1), axis=1, keepdims=True) + self.beta
        min_y = tf.reduce_min(tf.reduce_min(tf.abs(y_coords - self.beta), axis=1), axis=1, keepdims=True) + self.beta
        min_z = tf.reduce_min(tf.reduce_min(tf.abs(z_coords - self.beta), axis=1), axis=1, keepdims=True) + self.beta
        max_x = tf.reduce_max(tf.reduce_max(x_coords, axis=2), axis=1, keepdims=True)
        max_y = tf.reduce_max(tf.reduce_max(y_coords, axis=2), axis=1, keepdims=True)
        max_z = tf.reduce_max(tf.reduce_max(z_coords, axis=2), axis=1, keepdims=True)

        xyz_min = tf.expand_dims(tf.concat([min_x, min_y, min_z], axis=2), axis=-1)
        xyz_max = tf.expand_dims(tf.concat([max_x, max_y, max_z], axis=2), axis=-1)

        xyz_min = tf.math.minimum(xyz_min, xyz_max)

        features = self.map_from_to_opt(xyz_axis,
            xyz_min, xyz_max,
            tf.zeros_like(xyz_min), tf.ones_like(xyz_min)) * self.gamma + 1

        # sets all points which are not inside the current bounding volume to 0
        features = tf.einsum('bijf, bik -> bijf', features, mask)
        return (features, tf.squeeze(xyz_min), tf.squeeze(xyz_max))

    #@tf.function
    def call(self, input) :
        point_cloud, node_bounds = input

        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6] 

        node_mask = tf.stop_gradient(nss_tree_common.build_mask_EPO(point_cloud, node_bounds))

        # This code from here on implemnts Fig 1 of the paper
        # sets all points which are not inside the current bounding volume to 0
        point_cloud = tf.stop_gradient(tf.einsum('bij, bik -> bij', point_cloud, node_mask))
        # features are scaled to [0,1) and translated by +(1,1,1)
        # min_values are probably the minimal bounding box values before the scaling (x_min, y_min, z_min). Accordingly with max_values.
        features, min_values, max_values = self.object_normalize(point_cloud, node_mask)

        # the acutal convolution is performed here:
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        
        sum_per_dim = tf.reduce_sum(features, axis=1)
        denom = tf.math.divide_no_nan(1.0, tf.reduce_sum(node_mask, axis=1, keepdims=True))
        global_descr = denom * sum_per_dim
        global_descr = global_descr[:, tf.newaxis, :, :]

        # corresponds to Conv1Ds of Fig 1 of the paper
        local_thetas = self.offset_layer1(global_descr)
        local_thetas = self.offset_layer2(local_thetas)
        local_thetas = self.offset_layer3(local_thetas)
        local_thetas = self.flatten(local_thetas)

        x_split = tf.reduce_sum(tf.gather(local_thetas, indices=[0, 3, 6], axis=1), axis=1, keepdims=True) / 3.0
        y_split = tf.reduce_sum(tf.gather(local_thetas, indices=[1, 4, 7], axis=1), axis=1, keepdims=True) / 3.0
        z_split = tf.reduce_sum(tf.gather(local_thetas, indices=[2, 5, 8], axis=1), axis=1, keepdims=True) / 3.0
        local_thetas = tf.concat([x_split, y_split, z_split], axis=1) 

        s = tf.stop_gradient(tf.math.divide_no_nan(max_values - min_values, node_bmax - node_bmin))
        t = tf.stop_gradient(tf.math.divide_no_nan(min_values - node_bmin, node_bmax - node_bmin))

        return local_thetas, s, t