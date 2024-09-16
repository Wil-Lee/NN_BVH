# credits: https://github.com/cgaueb/nss
import tensorflow as tf
import nss_tree_common

class neural_spatial_node() :
    def __init__(self, name, lvl) :
        self.name = name
        self.lvl = lvl

        self.bounds = None
        self.mask = None
        
        self.theta = None
        self.parent_bounds = None
        self.parent_offset = None
        self.parent_normal = None
        self.parent_mask = None
       

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
import nss_tree_modules
@tf.function
def build_mask_EPO_(primitive_cloud, bmin, bmax, parent_offset, parent_normal, parent_mask, is_right_child, lower_bound = 0.0, upper_bound = 1.0):
    axis_points = nss_tree_modules.get_axis_points(primitive_cloud, parent_normal)

    mins = tf.reduce_min(axis_points, axis=2, keepdims=True)
    maxs = tf.reduce_max(axis_points, axis=2, keepdims=True)
    mids = mins + ((maxs - mins) * 0.5)

    if is_right_child:
        result = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(parent_offset[..., tf.newaxis] < mids, tf.float32))
    else:
        result = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(parent_offset[..., tf.newaxis] >= mids, tf.float32))

    return result

@tf.function
def build_mask_EPO(point_cloud, bounds, parent_offset, parent_normal, parent_mask, is_right_child, lower_bound = 0.0, upper_bound = 1.0):
    return build_mask_EPO_(point_cloud, bounds[:, 0:3], bounds[:, 3:], parent_offset[2], parent_normal, parent_mask, is_right_child, lower_bound, upper_bound)

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