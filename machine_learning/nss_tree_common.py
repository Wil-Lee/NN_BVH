# credits: https://github.com/cgaueb/nss
import tensorflow as tf

class neural_spatial_node() :
    def __init__(self, name, lvl) :
        self.name = name
        self.lvl = lvl

        self.bounds = None

        self.theta = None
        self.parent_bounds = None
        self.parent_offset = None
        self.parent_normal = None

@tf.function
def build_mask_(point_cloud, bmin, bmax, lower_bound = 0.0, upper_bound = 1.0) :
    # TODO: mask primitives here - every primitive not belonging to this node
    x = point_cloud[..., 0:1]
    y = point_cloud[..., 1:2]
    z = point_cloud[..., 2:3]

    x_bmin = bmin[..., 0:1]
    y_bmin = bmin[..., 1:2]
    z_bmin = bmin[..., 2:3]

    x_bmax = bmax[..., 0:1]
    y_bmax = bmax[..., 1:2]
    z_bmax = bmax[..., 2:3]

    mask_x = tf.where(tf.logical_and(
        x >= x_bmin[..., tf.newaxis],
        x <= x_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    mask_y = tf.where(tf.logical_and(
        y >= y_bmin[..., tf.newaxis],
        y <= y_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    mask_z = tf.where(tf.logical_and(
        z >= z_bmin[..., tf.newaxis],
        z <= z_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    # returns a bool vec whichs entries are 1 if the current node is inside the bound and 0 else
    return tf.math.multiply(tf.math.multiply(mask_x, mask_y), mask_z)

@tf.function
def build_mask(point_cloud, bounds, lower_bound = 0.0, upper_bound = 1.0) :
    return build_mask_(point_cloud, bounds[:, 0:3], bounds[:, 3:], lower_bound, upper_bound)

####################################################### EDIT ##################################################################################

@tf.function
def build_mask_EPO_(point_cloud, bmin, bmax, lower_bound = 0.0, upper_bound = 1.0):
    x1, x2, x3 = tf.unstack(point_cloud[..., :3], axis=-1)
    y1, y2, y3 = tf.unstack(point_cloud[..., 3:6], axis=-1) 
    z1, z2, z3 = tf.unstack(point_cloud[..., 6:9], axis=-1)

    x_bmin = bmin[..., 0:1]
    y_bmin = bmin[..., 1:2]
    z_bmin = bmin[..., 2:3]

    x_bmax = bmax[..., 0:1]
    y_bmax = bmax[..., 1:2]
    z_bmax = bmax[..., 2:3]

    mask_x = tf.where(tf.reduce_all(
        tf.stack([
            x1 >= x_bmin, x1 <= x_bmax,
            x2 >= x_bmin, x2 <= x_bmax,
            x3 >= x_bmin, x3 <= x_bmax
        ], axis=-1), axis=-1
    ), upper_bound, lower_bound)

    mask_y = tf.where(tf.reduce_all(
        tf.stack([
            y1 >= y_bmin, y1 <= y_bmax,
            y2 >= y_bmin, y2 <= y_bmax,
            y3 >= y_bmin, y3 <= y_bmax
        ], axis=-1), axis=-1
    ), upper_bound, lower_bound)

    mask_z = tf.where(tf.reduce_all(
        tf.stack([
            z1 >= z_bmin, z1 <= z_bmax,
            z2 >= z_bmin, z2 <= z_bmax,
            z3 >= z_bmin, z3 <= z_bmax
        ], axis=-1), axis=-1
    ), upper_bound, lower_bound)

    mask_x = tf.expand_dims(mask_x, axis=-1)
    mask_y = tf.expand_dims(mask_y, axis=-1)  
    mask_z = tf.expand_dims(mask_z, axis=-1)

    return tf.math.multiply(tf.math.multiply(mask_x, mask_y), mask_z)

@tf.function
def build_mask_EPO(point_cloud, bounds, lower_bound = 0.0, upper_bound = 1.0):
    return build_mask_EPO_(point_cloud, bounds[:, 0:3], bounds[:, 3:], lower_bound, upper_bound)

###############################################################################################################################################
@tf.function
def build_mask_1D(point_cloud, bmin, bmax) :
    return tf.where(tf.logical_and(
        point_cloud >= bmin[..., tf.newaxis],
        point_cloud <= bmax[..., tf.newaxis]), 1.0, 0.0)

@tf.function
def build_mask1D(point_cloud, bounds) :
    return build_mask_1D(point_cloud, bounds[:, 0:], bounds[:, 1:])

@tf.function
def N_fn(point_clouds, bounds) :
    parent_mask = build_mask(point_clouds, bounds)
    N = tf.reduce_sum(parent_mask, axis=1)
    return N