# credits: https://github.com/cgaueb/nss
import tensorflow as tf
import nss_tree_common

class neuralNode_splitter(tf.Module) :
    def __init__(self, config) :
        super(neuralNode_splitter, self).__init__()
        self.config = config
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        self.gen_fn = self.gen_nodes_EPO if config['EPO'] else self.gen_nodes
        self.beta = config['beta']

    @tf.function
    def tight_bounds(self, point_clouds, node_bounds) :
        mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, node_bounds))
        masked_pc = tf.stop_gradient(tf.einsum('bij, bik -> bij', point_clouds, mask))
        N = tf.reduce_sum(mask, axis=1)
        xyz_axis = tf.expand_dims(masked_pc, axis=-1)
        xyz_min = tf.reduce_min(tf.abs(xyz_axis - 1.0), axis=1, keepdims=True) + 1.0
        xyz_max = tf.reduce_max(xyz_axis, axis=1, keepdims=True)
        xyz_min = tf.math.minimum(xyz_min, xyz_max)
        tight_node_bmin = tf.squeeze(xyz_min)
        tight_node_bmax = tf.squeeze(xyz_max)
        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]
        diag = node_bmax - node_bmin
        theta_bmin = tf.stop_gradient(tf.math.divide_no_nan(tight_node_bmin - node_bmin, diag))
        theta_bmax = tf.stop_gradient(tf.math.divide_no_nan(tight_node_bmax - node_bmin, diag))

        return \
            tf.where(tf.greater(N, 0), node_bmin + theta_bmin * diag, node_bmin), \
            tf.where(tf.greater(N, 0), node_bmin + theta_bmax * diag, node_bmax)

    @tf.function
    def gen_nodes_EPO(self, normal, theta, node_bounds, point_clouds):
        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]

        b0 = tf.einsum('bk, k -> b', node_bmin, normal)[..., tf.newaxis]
        b1 = tf.einsum('bk, k -> b', node_bmax, normal)[..., tf.newaxis]
        theta_offset = b0 + theta * (b1 - b0)

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask_EPO(point_clouds, node_bounds))

        if tf.reduce_all(tf.equal(normal, [1,0,0])):
            axis_points = point_clouds[:,:,0:3]
        elif tf.reduce_all(tf.equal(normal, [0,1,0])):
            axis_points = point_clouds[:,:,3:6]
        else:
            axis_points = point_clouds[:,:,6:9]

        # TODO: if network doesn't behave as aspected - these bounds should be investigated
        (theta_offset_left, theta_offset_right) = child_bounds(self.beta, axis_points, parent_mask, b0, b1, theta)

        right_bmin_temp = tf.einsum('bk, k -> bk', node_bmin, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset_right, normal)
        right_bmax_temp = node_bmax

        right_bmin = tf.minimum(right_bmin_temp, right_bmax_temp)
        right_bmax = tf.maximum(right_bmin_temp, right_bmax_temp)

        left_bmin_temp = node_bmin
        left_bmax_temp = tf.einsum('bk, k -> bk', node_bmax, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset_left, normal)

        left_bmin = tf.minimum(left_bmin_temp, left_bmax_temp)
        left_bmax = tf.maximum(left_bmin_temp, left_bmax_temp)

        left_bbox = tf.concat([left_bmin, left_bmax], axis=-1)
        right_bbox = tf.concat([right_bmin, right_bmax], axis=-1)

        return (theta_offset_left, theta_offset_right, theta_offset), left_bbox, right_bbox

    @tf.function
    def gen_nodes(self, normal, theta, node_bounds, point_clouds) :
        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]

        b0 = tf.einsum('bk, k -> b', node_bmin, normal)[..., tf.newaxis]
        b1 = tf.einsum('bk, k -> b', node_bmax, normal)[..., tf.newaxis]
        # global offset
        theta_offset = b0 + theta * (b1 - b0)

        right_bmin_temp = tf.einsum('bk, k -> bk', node_bmin, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset, normal)
        right_bmax_temp = node_bmax

        right_bmin = tf.minimum(right_bmin_temp, right_bmax_temp)
        right_bmax = tf.maximum(right_bmin_temp, right_bmax_temp)

        left_bmin_temp = node_bmin
        left_bmax_temp = tf.einsum('bk, k -> bk', node_bmax, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset, normal)

        left_bmin = tf.minimum(left_bmin_temp, left_bmax_temp)
        left_bmax = tf.maximum(left_bmin_temp, left_bmax_temp)

        left_bbox = tf.concat([left_bmin, left_bmax], axis=-1)
        right_bbox = tf.concat([right_bmin, right_bmax], axis=-1)
        
        return theta_offset, left_bbox, right_bbox

    @tf.function
    def __call__(self, node_bounds, thetas, point_clouds=None) :
        thetas_X = thetas[:, 0:1]
        thetas_Y = thetas[:, 1:2]
        thetas_Z = thetas[:, 2:3]

        offset_X, left_bbox_X, right_bbox_X = self.gen_fn(self.nX, thetas_X, node_bounds, point_clouds)
        offset_Y, left_bbox_Y, right_bbox_Y = self.gen_fn(self.nY, thetas_Y, node_bounds, point_clouds)
        offset_Z, left_bbox_Z, right_bbox_Z = self.gen_fn(self.nZ, thetas_Z, node_bounds, point_clouds)

        return (offset_X, offset_Y, offset_Z), \
            (left_bbox_X, right_bbox_X, left_bbox_Y, right_bbox_Y, left_bbox_Z, right_bbox_Z)
    
####################################################### EDIT ##################################################################################

@tf.function
@tf.custom_gradient
def child_bounds(beta, axis_points, parent_mask, parent_min, parent_max, offset):
    offset = offset[..., tf.newaxis] #* 0 + 1.5 # debugging
    mins = tf.reduce_min(axis_points, axis=2, keepdims=True)
    maxs = tf.reduce_max(axis_points, axis=2, keepdims=True)
    
    prims_left_to_split_mask = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(maxs < offset, tf.float32))
    prims_isecting_split_mask = tf.einsum('bij, bij -> bij', parent_mask, \
                                                tf.cast(tf.logical_and(mins <= offset, offset <= maxs), tf.float32))
    left_child_prims_isecting_split_mask = tf.einsum('bij, bij -> bij', prims_isecting_split_mask, \
        tf.cast(tf.abs(offset - mins) >= tf.abs(maxs - offset), tf.float32))
    left_child_max_bound = tf.reduce_max(tf.einsum('bij, bij -> bij', maxs, left_child_prims_isecting_split_mask), axis=1)
    
    left_child_prims_bool_mask = tf.cast(left_child_prims_isecting_split_mask + prims_left_to_split_mask, tf.bool)
    right_child_prims_mask = tf.cast(tf.logical_not(left_child_prims_bool_mask), tf.float32)
    total_max = tf.reduce_max(maxs)
    # needed to eliminate 0s for reduce_min by adding total_max to all 0s of right child's primitives
    right_child_prims_non_zero = (tf.cast(left_child_prims_bool_mask, tf.float32) * total_max) \
        + tf.einsum('bij, bij -> bij', mins, right_child_prims_mask)
    right_child_min_bound = tf.reduce_min(right_child_prims_non_zero, axis=1)
    
    N = tf.reduce_sum(tf.cast(left_child_prims_bool_mask, tf.float32), axis=1)

    @tf.function
    def next_step(mask, split_value):
        right_child_mins = tf.einsum('bij, bij -> bij', mins, right_child_prims_mask)
        right_child_maxs = tf.einsum('bij, bij -> bij', maxs, right_child_prims_mask)

        right_child_prims_mids = right_child_mins + ((right_child_maxs - right_child_mins) * 0.5)
        right_child_prims_mids_non_zero = (tf.cast(left_child_prims_bool_mask, tf.float32) * total_max) \
            + right_child_prims_mids
        
        offset_above = tf.reduce_min(right_child_prims_mids_non_zero, axis=1, keepdims=True)
        prims_increase = tf.reduce_sum(tf.cast(tf.equal(right_child_prims_mids, offset_above), tf.float32), axis=1)
        axis_max = tf.reduce_max(maxs, axis=1, keepdims=True)
        offset_above = tf.math.minimum(offset_above, axis_max)

        N1 = N + prims_increase
        return offset_above, N1
    
    @tf.function
    def grad(upstream, _):
        offset_above, N1 = next_step(parent_mask, offset)
        slope = tf.math.divide_no_nan(N1 - N, offset_above[..., 0] - offset[..., 0])
        stepGrad = tf.clip_by_value(slope, 0.0, 1.0 / 0.0001)

        upstream_grad = stepGrad
        upstream_grad = tf.einsum('bi, bi -> bi', upstream, upstream_grad)
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] >= parent_min, tf.float32))
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] <= parent_max, tf.float32))
        return None, None, None, None, None, upstream_grad

    return (left_child_max_bound, right_child_min_bound), grad
###############################################################################################################################################


class sah_eval(tf.Module) :
    """ Returns the probability of a ray that intersects the parent node also intersects the current node. """
    def __init__(self) :
        super(sah_eval, self).__init__()

    @tf.function
    def area(self, bounds) :
        bmin = bounds[:, 0:3]
        bmax = bounds[:, 3:6]
        diag = bmax - bmin
        x = diag[:, 0:1]
        y = diag[:, 1:2]
        z = diag[:, 2:3]
        return 2.0 * (x * y + x * z + y * z)

    @tf.function
    def __call__(self, parent_bounds, bounds, point_clouds) :
        parent_area = self.area(parent_bounds)
        self_area = self.area(bounds)
        return self_area / parent_area

class vh_eval(tf.Module) :
    def __init__(self, r_eps=1.e-4) :
        super(vh_eval, self).__init__()
        self.r = r_eps

    @tf.function
    def volume(self, bounds) :
        bmin = bounds[:, 0:3] - self.r
        bmax = bounds[:, 3:6] + self.r
        diag = bmax - bmin
        x = diag[:, 0:1]
        y = diag[:, 1:2]
        z = diag[:, 2:3]
        return (x * y * z)

    @tf.function
    def __call__(self, parent_bounds, bounds, point_clouds) :
        parent_vol = self.volume(parent_bounds)
        self_vol = self.volume(bounds)
        return self_vol / parent_vol

class gr_q_eval(tf.Module) :
    def __init__(self, t) :
        super(gr_q_eval, self).__init__()
        self.t_cost = t

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, bounds) :
        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, bounds))
        N = tf.reduce_sum(parent_mask, axis=1)
        return N * self.t_cost

class q_eval(tf.Module) :
    """ Returns the cost of traversal and intersection tests of the left and right child node. C(L), C(R)"""
    def __init__(self, t, beta) :
        super(q_eval, self).__init__()
        self.t_cost = t
        self.count_fn = qL_fn
        self.beta = beta

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask) :
        axis_points = tf.einsum('bijk, j -> bik', point_clouds[..., tf.newaxis], parent_normal)
        parent_minmax = tf.einsum('bij, j -> bi', tf.reshape(parent_bounds, [-1, 2, 3]), parent_normal)
        N = tf.stop_gradient(tf.reduce_sum(parent_mask, axis=1))
        nL = self.count_fn(self.beta, axis_points, parent_mask, parent_minmax[..., 0:1], parent_minmax[..., 1:], parent_offset)
        return nL * self.t_cost, (N - nL) * self.t_cost

class p_eval(tf.Module) :
    """ Just returns the traversal cost coefficient. """
    def __init__(self, t) :
        super(p_eval, self).__init__()
        self.t_cost = t

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, bounds) :
        return self.t_cost

@tf.function
@tf.custom_gradient
def qL_fn(beta, axis_points, parent_mask, parent_min, parent_max, offset) :
    # axis points are the projection of the points of the pc to the splitting dimension: split on x-axis -> projection to x
    offset = offset[..., tf.newaxis]
    node_mask = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(axis_points <= offset, tf.float32))
    N = tf.reduce_sum(node_mask, axis=1)

    @tf.function
    def next_step(mask, b):
        # create a mask for the offset (left of b = 0, right of b = 1)
        mask_after_offset = tf.einsum('bij, bij -> bij', mask, tf.cast(axis_points > b, tf.float32))
        # masks out all points which lay left of b (split_offset)
        axisR = tf.einsum('bij, bij -> bij', axis_points, mask_after_offset)
        # returns the next split_offset where the amount of points in the left child gets incremented by one
        offset_above = tf.reduce_min(tf.abs(axisR - beta), axis=1, keepdims=True) + beta
        # extracts the max point value of a right child in the current split dimension
        axis_max = tf.reduce_max(axisR, axis=1, keepdims=True)
        # edge case handling
        offset_above = tf.math.minimum(offset_above, axis_max)
        # probably means N+1 because it returns the amount of points left to the split_offset + 1 (the next offset value where N increases by one = offset_above)
        # it is possible that the difference between N and N1 is bigger than 1 because there could be multiple points at offset_above
        N1 = tf.reduce_sum(tf.einsum('bij, bij -> bij', mask, tf.cast(axis_points <= offset_above, tf.float32)), axis=1)
        return offset_above, N1

    @tf.function
    def grad(upstream):
        offset_above, N1 = next_step(parent_mask, offset)
        # difference quotient between the current offset and the next offset where the size of the left child points increases
        slope = tf.math.divide_no_nan(N1 - N, offset_above[..., 0] - offset[..., 0])
        # clips the difference quotient between the current offset and the next offset where the size of the left child points increases
        stepGrad = tf.clip_by_value(slope, 0.0, 1.0 / 0.0001)

        upstream_grad = stepGrad
        # combines the local gradient with the upstream gradient
        upstream_grad = tf.einsum('bi, bi -> bi', upstream, upstream_grad)
        # masks out all upstream values with invalid offset
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] >= parent_min, tf.float32))
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] <= parent_max, tf.float32))
        return None, None, None, None, None, upstream_grad

    return N, grad


@tf.function
@tf.custom_gradient
def soft_min4(v0, v1, v2, v3, t) :
    vals = tf.concat([v0, v1, v2, v3], axis=-1)
    ret = tf.reduce_min(vals, axis=-1, keepdims=True)
    
    @tf.function
    def grad(upstream) :
        x = -t * vals
        x -= tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.nn.softmax(x, axis=-1)
        upstream_grad = upstream * x
        return upstream_grad[:, 0:1], upstream_grad[:, 1:2], upstream_grad[:, 2:3], upstream_grad[:, 3:4], None

    return ret, grad

@tf.function
@tf.custom_gradient
def soft_min3(v0, v1, v2, v3, t) :
    vals = tf.concat([v0, v1, v2], axis=-1)
    ret = tf.reduce_min(vals, axis=-1, keepdims=True)

    @tf.function
    def grad(upstream) :
        x = -t * vals
        x -= tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.nn.softmax(x, axis=-1)

        upstream_grad = upstream * x
        return upstream_grad[:, 0:1], upstream_grad[:, 1:2], upstream_grad[:, 2:3], None, None
    
    return ret, grad

@tf.function
def hard_min4(v0, v1, v2, v3) :
    vals = tf.concat([v0, v1, v2, v3], axis=-1)
    return vals, tf.reduce_min(vals, axis=-1, keepdims=True)

@tf.function
def hard_min3(v0, v1, v2, v3) :
    vals = tf.concat([v0, v1, v2], axis=-1)
    return vals, tf.reduce_min(vals, axis=-1, keepdims=True)



class pool_treelet(tf.Module) :
    def __init__(self, t, num_splits, p_eval, q_eval, gr_q_eval, w_eval, normFactor) :
        super(pool_treelet, self).__init__()
        self.t = t
        self.num_splits = num_splits
        self.soft_min = soft_min3 if num_splits == 3 else soft_min4
        self.hard_min = hard_min3 if num_splits == 3 else hard_min4
        self.p_eval = p_eval
        self.q_eval = q_eval
        self.w_eval = w_eval
        self.gr_q_eval = gr_q_eval
        self.normFactor = 1.0 / normFactor
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        self.isect_cost = self.q_eval.t_cost
        self.beta = self.q_eval.beta

    @tf.function
    def get_pred_branch_from_leaves(self, cost_xyz, offsets_xyz) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)
        return pred_planes[:, tf.newaxis, :]

    @tf.function
    def get_pred_branch_from_interior(self, cost_xyz, offsets_xyz, subtree_splits) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)[:, tf.newaxis, :]
        pred_splits = tf.gather(subtree_splits, pred_axis, axis=1, batch_dims=1)
        pred_planes = tf.concat([pred_planes, pred_splits], axis=1)
        return pred_planes

    @tf.function
    def eval_leaves(self, point_clouds,
        # q_eval returns the amount of leaves for both nodes
        # p_eval returns the traversal cost constant
        # w_eval the heuristic specific cost e.g. SAH: surface of the node devided by the root node's surface
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
        flag) :

        Cnode = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval(root_bounds, node_bounds, point_clouds)

        node_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, node_bounds))
        
        qL_X, qR_X = self.q_eval(point_clouds, self.nX, offsetX, node_bounds, node_mask)
        qL_Y, qR_Y = self.q_eval(point_clouds, self.nY, offsetY, node_bounds, node_mask)
        qL_Z, qR_Z = self.q_eval(point_clouds, self.nZ, offsetZ, node_bounds, node_mask)
        
        CxL = qL_X * self.w_eval(root_bounds, xL_bounds, point_clouds)
        CxR = qR_X * self.w_eval(root_bounds, xR_bounds, point_clouds)
        CyL = qL_Y * self.w_eval(root_bounds, yL_bounds, point_clouds)
        CyR = qR_Y * self.w_eval(root_bounds, yR_bounds, point_clouds)
        CzL = qL_Z * self.w_eval(root_bounds, zL_bounds, point_clouds)
        CzR = qR_Z * self.w_eval(root_bounds, zR_bounds, point_clouds)

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.zeros_like(qL_X) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)
        
        return Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf

    @tf.function
    def eval_interior(self, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds) :
        
        Cnode = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval(root_bounds, node_bounds, point_clouds)

        return Cnode

    @tf.function
    def pool_leaves_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)
        
        return (self.pool_soft(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf), Cleaf)
        
    
    @tf.function
    def pool_leaves(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)
    
        return (Cnode + CxL + CxR, Cnode + CyL + CyR, Cnode + CzL + CzR)

    @tf.function
    def pool_interior_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        CxL, CxR, CyL, CyR, CzL, CzR,
        CxL_leaf, CxR_leaf,
        CyL_leaf, CyR_leaf,
        CzL_leaf, CzR_leaf) :
        
        Cnode = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)
        
        return self.pool_soft(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf)

    @tf.function
    def pool_soft(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        treelet_cost = self.soft_min(cost_x, cost_y, cost_z, Cleaf, self.t)
        return treelet_cost
    
    @tf.function
    def pool_leaves_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)

        return self.pool_structure_leaves(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ)

    @tf.function
    def pool_interior_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        offsetX, offsetY, offsetZ) :
        
        Cnode = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)

        return self.pool_structure_interior(Cnode,
            branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
            Cleaf, offsetX, offsetY, offsetZ)

    @tf.function
    def pool_structure_leaves(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)
        return min_cost, \
            self.get_pred_branch_from_leaves(cost_xyz,
                tf.concat([offsetX, offsetY, offsetZ, tf.ones_like(offsetZ)], axis=-1))

    @tf.function
    def pool_structure_interior(self, Cnode,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        Cleaf, offsetX, offsetY, offsetZ) :

        CxL, planes_xL = branch_xL
        CxR, planes_xR = branch_xR
        CyL, planes_yL = branch_yL
        CyR, planes_yR = branch_yR
        CzL, planes_zL = branch_zL
        CzR, planes_zR = branch_zR

        plane_x = tf.concat([planes_xL, planes_xR], axis=1)
        plane_y = tf.concat([planes_yL, planes_yR], axis=1)
        plane_z = tf.concat([planes_zL, planes_zR], axis=1)

        split_planes = tf.concat([
            plane_x[:, tf.newaxis, ...],
            plane_y[:, tf.newaxis, ...],
            plane_z[:, tf.newaxis, ...],
            ], axis=1)

        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR

        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)

        return min_cost, \
            self.get_pred_branch_from_interior(cost_xyz,
                tf.concat([offsetX, offsetY, offsetZ, tf.ones_like(offsetZ)], axis=-1),
                split_planes)


####################################################### EDIT ##################################################################################

@tf.function
@tf.custom_gradient
def qL_fn_SAH(beta, axis_points, parent_mask, parent_min, parent_max, offset):
    left_offset = offset[0][..., tf.newaxis]
    right_offset = offset[1][..., tf.newaxis]
    
    mins = tf.reduce_min(axis_points, axis=2, keepdims=True)
    maxs = tf.reduce_max(axis_points, axis=2, keepdims=True)
    
    left_child_prims_mask = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(maxs <= left_offset, tf.float32))
    right_child_prims_mask = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(mins >= right_offset, tf.float32))
    
    total_max = tf.reduce_max(maxs)
    
    N = tf.reduce_sum(left_child_prims_mask, axis=1)
    
    @tf.function
    def next_step(mask):
        right_child_mins = tf.einsum('bij, bij -> bij', mins, right_child_prims_mask)
        right_child_maxs = tf.einsum('bij, bij -> bij', maxs, right_child_prims_mask)

        right_child_prims_mids = right_child_mins + ((right_child_maxs - right_child_mins) * 0.5)
        right_child_prims_mids_non_zero = (left_child_prims_mask * total_max) \
            + right_child_prims_mids

        offset_above = tf.reduce_min(right_child_prims_mids_non_zero, axis=1, keepdims=True)
        prims_increase = tf.reduce_sum(tf.cast(tf.equal(right_child_prims_mids, offset_above), tf.float32), axis=1)
        axis_max = tf.reduce_max(maxs, axis=1, keepdims=True)
        offset_above = tf.math.minimum(offset_above, axis_max)

        N1 = N + prims_increase
        return offset_above, N1

    @tf.function
    def grad(upstream):
        offset_above, N1 = next_step(parent_mask)
        split_offset = (offset[2])[..., tf.newaxis]
        slope = tf.math.divide_no_nan(N1 - N, offset_above[..., 0] - split_offset[..., 0])
        stepGrad = tf.clip_by_value(slope, 0.0, 1.0 / 0.0001)

        upstream_grad = stepGrad
        upstream_grad = tf.einsum('bi, bi -> bi', upstream, upstream_grad)
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(split_offset[..., 0] >= parent_min, tf.float32))
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(split_offset[..., 0] <= parent_max, tf.float32))
        return None, None, None, None, None, None, None, upstream_grad
    
    return N, grad


class pool_treelet_EPO(tf.Module) :
    def __init__(self, t, t_iscet_cost, num_splits, normFactor, beta, i_isect_cost, epo_sah_alpha) :
        super(pool_treelet_EPO, self).__init__()
        self.t = t
        self.t_isect_cost = t_iscet_cost
        self.num_splits = num_splits
        self.soft_min = soft_min3 if num_splits == 3 else soft_min4
        self.hard_min = hard_min3 if num_splits == 3 else hard_min4
        self.normFactor = 1.0 / normFactor
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        self.i_isect_cost = i_isect_cost
        self.beta = beta
        self.w_eval_SAH = sah_eval()
        self.wL_fn_EPO = wL_fn_EPO
        self.qL_fn_SAH = qL_fn_SAH
        self.EPO_SAH_alpha = epo_sah_alpha

    @tf.function
    def get_pred_branch_from_leaves(self, cost_xyz, offsets_xyz) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)
        return pred_planes[:, tf.newaxis, :]

    @tf.function
    def get_pred_branch_from_interior(self, cost_xyz, offsets_xyz, subtree_splits) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)[:, tf.newaxis, :]
        pred_splits = tf.gather(subtree_splits, pred_axis, axis=1, batch_dims=1)
        pred_planes = tf.concat([pred_planes, pred_splits], axis=1)
        return pred_planes
    
    @tf.function
    def eval_leaves_EPO(self, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
        flag) :

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))

        Cnode_EPO = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask)
        
        Cnode_SAH = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval_SAH(root_bounds, node_bounds, point_clouds)

        node_mask = tf.stop_gradient(nss_tree_common.build_mask_EPO(point_clouds, node_bounds))
        
        qL_X, qR_X = self.q_eval(point_clouds, self.nX, offsetX, node_bounds, node_mask)
        qL_Y, qR_Y = self.q_eval(point_clouds, self.nY, offsetY, node_bounds, node_mask)
        qL_Z, qR_Z = self.q_eval(point_clouds, self.nZ, offsetZ, node_bounds, node_mask)
        
        CxL_SAH = qL_X * self.w_eval_SAH(root_bounds, xL_bounds, point_clouds)
        CxR_SAH = qR_X * self.w_eval_SAH(root_bounds, xR_bounds, point_clouds)
        CyL_SAH = qL_Y * self.w_eval_SAH(root_bounds, yL_bounds, point_clouds)
        CyR_SAH = qR_Y * self.w_eval_SAH(root_bounds, yR_bounds, point_clouds)
        CzL_SAH = qL_Z * self.w_eval_SAH(root_bounds, zL_bounds, point_clouds)
        CzR_SAH = qR_Z * self.w_eval_SAH(root_bounds, zR_bounds, point_clouds)
        
        QleafL_SAH, QleafR_SAH = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot_SAH = tf.zeros_like(qL_X) * self.normFactor
        Qleaf_SAH = QleafRoot_SAH * flag[0] + QleafL_SAH * flag[1] + QleafR_SAH * flag[2]
        Cleaf = Qleaf_SAH * self.w_eval_SAH(root_bounds, node_bounds, point_clouds) # only needed for unbalanced tree
        
        CxL_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qL_X
        CxR_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qR_X
        CyL_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qL_Y
        CyR_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qR_Y
        CzL_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qL_Z 
        CzR_EPO = self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask) # * qR_Z
        #Cleaf_EPO = (Cnode_EPO / self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds)) * self.t_isect_cost


        Cnode = (1 - self.EPO_SAH_alpha) * Cnode_SAH + self.EPO_SAH_alpha * Cnode_EPO
        CxL = (1 - self.EPO_SAH_alpha) * CxL_SAH + self.EPO_SAH_alpha * CxL_EPO
        CxR = (1 - self.EPO_SAH_alpha) * CxR_SAH + self.EPO_SAH_alpha * CxR_EPO
        CyL = (1 - self.EPO_SAH_alpha) * CyL_SAH + self.EPO_SAH_alpha * CyL_EPO
        CyR = (1 - self.EPO_SAH_alpha) * CyR_SAH + self.EPO_SAH_alpha * CyR_EPO
        CzL = (1 - self.EPO_SAH_alpha) * CzL_SAH + self.EPO_SAH_alpha * CzL_EPO
        CzR = (1 - self.EPO_SAH_alpha) * CzR_SAH + self.EPO_SAH_alpha* CzR_EPO
        #Cleaf = (1 - scene_alpha) * Cleaf_SAH + scene_alpha * Cleaf_EPO
        
        return Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf

    @tf.function
    def eval_interior(self, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds) :

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        
        Cnode_EPO = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval_EPO(point_clouds, node_bounds, parent_normal, parent_bounds, parent_mask)
        
        Cnode_SAH = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval_SAH(root_bounds, node_bounds, point_clouds)

        return Cnode_SAH, Cnode_EPO
    
    @tf.function
    def pool_leaves_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves_EPO(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)
        
        return (self.pool_soft_EPO(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf), Cleaf)
    
    @tf.function
    def pool_interior_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        CxL, CxR, CyL, CyR, CzL, CzR,
        CxL_leaf, CxR_leaf,
        CyL_leaf, CyR_leaf,
        CzL_leaf, CzR_leaf) :
        
        Cnode_SAH, Cnode_EPO = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)
        
        Cnode = (1 - self.EPO_SAH_alpha) * Cnode_SAH + self.EPO_SAH_alpha * Cnode_EPO

        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2] 
        Cleaf = Qleaf * self.w_eval_SAH(root_bounds, node_bounds, point_clouds) # only needed for unbalanced tree
        
        return self.pool_soft_EPO(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf)
    
    @tf.function
    def pool_soft_EPO(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        treelet_cost = self.soft_min(cost_x, cost_y, cost_z, Cleaf, self.t)
        return treelet_cost

    @tf.function
    def pool_leaves_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves_EPO(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)

        return self.pool_structure_leaves(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ)
    
    @tf.function
    def pool_interior_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        offsetX, offsetY, offsetZ) :
        
        Cnode_SAH, Cnode_EPO = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)
        
        Cnode = (1 - self.EPO_SAH_alpha) * Cnode_SAH + self.EPO_SAH_alpha * Cnode_EPO


        parent_mask = tf.stop_gradient(nss_tree_common.build_mask(point_clouds, parent_bounds))
        if len(parent_offset) == 1:
            QleafL = tf.cast(tf.shape(point_clouds)[1], tf.float32)
            QleafR = 0.0
        else:
            QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval_SAH(root_bounds, node_bounds, point_clouds)

        return self.pool_structure_interior(Cnode,
            branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
            Cleaf, offsetX, offsetY, offsetZ)
    
    @tf.function
    def pool_structure_leaves(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)
        return min_cost, \
            self.get_pred_branch_from_leaves(cost_xyz,
                tf.concat([offsetX[2], offsetY[2], offsetZ[2], tf.ones_like(offsetZ[2])], axis=-1))
    
    @tf.function
    def pool_structure_interior(self, Cnode,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        Cleaf, offsetX, offsetY, offsetZ) :

        CxL, planes_xL = branch_xL
        CxR, planes_xR = branch_xR
        CyL, planes_yL = branch_yL
        CyR, planes_yR = branch_yR
        CzL, planes_zL = branch_zL
        CzR, planes_zR = branch_zR

        plane_x = tf.concat([planes_xL, planes_xR], axis=1)
        plane_y = tf.concat([planes_yL, planes_yR], axis=1)
        plane_z = tf.concat([planes_zL, planes_zR], axis=1)

        split_planes = tf.concat([
            plane_x[:, tf.newaxis, ...],
            plane_y[:, tf.newaxis, ...],
            plane_z[:, tf.newaxis, ...],
            ], axis=1)

        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        # Cleaf is ignored
        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)

        return min_cost, \
            self.get_pred_branch_from_interior(cost_xyz,
                tf.concat([offsetX[2], offsetY[2], offsetZ[2], tf.ones_like(offsetZ[2])], axis=-1),
                split_planes)

    @tf.function
    def p_eval(self, point_clouds, parent_normal, parent_offset, parent_bounds, bounds) :
        return self.i_isect_cost
    
    @tf.function
    def get_axis_points(self, primitive_cloud, parent_normal):
        parent_normal_expanded = tf.repeat(parent_normal, repeats=3)
        masked_primitives = tf.multiply(primitive_cloud, parent_normal_expanded)

        # always selects one of the addends -> x1,x2,x3 or y1,y2,y3 or z1,z2,z3
        p1 = masked_primitives[:,:,0:1] + masked_primitives[:,:,3:4] + masked_primitives[:,:,6:7]
        p2 = masked_primitives[:,:,1:2] + masked_primitives[:,:,4:5] + masked_primitives[:,:,7:8]
        p3 = masked_primitives[:,:,2:3] + masked_primitives[:,:,5:6] + masked_primitives[:,:,8:9]

        return tf.concat([p1, p2, p3], axis=-1)
    
    @tf.function
    def q_eval(self, primitive_cloud, parent_normal, parent_offset, parent_bounds, parent_mask):
        axis_points = self.get_axis_points(primitive_cloud, parent_normal)      

        parent_minmax = tf.einsum('bij, j -> bi', tf.reshape(parent_bounds, [-1, 2, 3]), parent_normal)
        N = tf.stop_gradient(tf.reduce_sum(parent_mask, axis=1))
        nL = self.qL_fn_SAH(self.beta, axis_points, parent_mask, parent_minmax[..., 0:1], parent_minmax[..., 1:], parent_offset)[0]
        return nL * self.t_isect_cost, (N - nL) * self.t_isect_cost
    
    @tf.function
    def w_eval_EPO(self, primitive_cloud, node_bounds, parent_normal, parent_bounds, parent_mask):
        axis_points = self.get_axis_points(primitive_cloud, parent_normal)

        parent_minmax = tf.einsum('bij, j -> bi', tf.reshape(parent_bounds, [-1, 2, 3]), parent_normal)
        node_minmax = tf.einsum('bij, j -> bi', tf.reshape(node_bounds, [-1, 2, 3]), parent_normal)

        return self.wL_fn_EPO(
            node_bounds=node_bounds,
            node_min=node_minmax[..., 0:1],
            node_max=node_minmax[..., 1:2],
            beta=self.beta, 
            axis_points=axis_points, 
            parent_mask=parent_mask,  
            parent_min=parent_minmax[..., 0:1],
            parent_max=parent_minmax[..., 1:2],
            primitive_cloud=primitive_cloud)[0]
    

@tf.function
def get_prims_intersecting_node_mask(node_bounds, parent_mask, primitive_cloud):
    """ Returns the primtives which intersect the given node bounds. """
    x1, x2, x3 = tf.unstack(primitive_cloud[..., :3], axis=-1)
    y1, y2, y3 = tf.unstack(primitive_cloud[..., 3:6], axis=-1) 
    z1, z2, z3 = tf.unstack(primitive_cloud[..., 6:9], axis=-1)

    x_bmin = node_bounds[..., 0:1]
    y_bmin = node_bounds[..., 1:2]
    z_bmin = node_bounds[..., 2:3]
    x_bmax = node_bounds[..., 3:4]
    y_bmax = node_bounds[..., 4:5]
    z_bmax = node_bounds[..., 5:6]

    x1_in_bounds = tf.logical_and(x1 >= x_bmin, x1 <= x_bmax)
    x2_in_bounds = tf.logical_and(x2 >= x_bmin, x2 <= x_bmax)
    x3_in_bounds = tf.logical_and(x3 >= x_bmin, x3 <= x_bmax)

    y1_in_bounds = tf.logical_and(y1 >= y_bmin, y1 <= y_bmax)
    y2_in_bounds = tf.logical_and(y2 >= y_bmin, y2 <= y_bmax)
    y3_in_bounds = tf.logical_and(y3 >= y_bmin, y3 <= y_bmax)

    z1_in_bounds = tf.logical_and(z1 >= z_bmin, z1 <= z_bmax)
    z2_in_bounds = tf.logical_and(z2 >= z_bmin, z2 <= z_bmax)
    z3_in_bounds = tf.logical_and(z3 >= z_bmin, z3 <= z_bmax)

    point1_in_bounds = tf.logical_and(tf.logical_and(x1_in_bounds, y1_in_bounds), z1_in_bounds)
    point2_in_bounds = tf.logical_and(tf.logical_and(x2_in_bounds, y2_in_bounds), z2_in_bounds)
    point3_in_bounds = tf.logical_and(tf.logical_and(x3_in_bounds, y3_in_bounds), z3_in_bounds)

    at_least_one_point_inside = tf.logical_or(tf.logical_or(point1_in_bounds, point2_in_bounds), point3_in_bounds)
    at_least_one_point_outside = tf.logical_not(tf.logical_and(tf.logical_and(point1_in_bounds, point2_in_bounds), point3_in_bounds))

    total_prims_intersecting_node = tf.cast(tf.expand_dims(tf.logical_and(at_least_one_point_inside, at_least_one_point_outside), axis=-1), tf.float32)

    prims_in_sibling_node_mask = tf.multiply(total_prims_intersecting_node, parent_mask)
    prims_outside_sibling_intersecting_node_mask = total_prims_intersecting_node - prims_in_sibling_node_mask 

    points_inside_node_mask = tf.concat([tf.cast(point1_in_bounds[..., tf.newaxis], tf.float32), \
                                            tf.cast(point2_in_bounds[..., tf.newaxis], tf.float32), \
                                            tf.cast(point3_in_bounds[..., tf.newaxis], tf.float32)], axis=-1)
    
    # what a name ;)
    prims_outside_sibling_intersecting_node_inside_points = tf.einsum('bij, bik -> bij', points_inside_node_mask,\
                                                                        prims_outside_sibling_intersecting_node_mask)

    return total_prims_intersecting_node, prims_in_sibling_node_mask, prims_outside_sibling_intersecting_node_inside_points 

@tf.function
def surface_prims_EPO(prims):
    P1 = tf.stack([prims[:, :, 0], prims[:, :, 3], prims[:, :, 6]], axis=-1)
    P2 = tf.stack([prims[:, :, 1], prims[:, :, 4], prims[:, :, 7]], axis=-1)
    P3 = tf.stack([prims[:, :, 2], prims[:, :, 5], prims[:, :, 8]], axis=-1)
    
    AB = P2 - P1
    AC = P3 - P1
    
    u = tf.linalg.cross(AB, AC)
    prims_surfaces = tf.expand_dims(tf.norm(u, axis=-1), axis=-1)
    surface_area = tf.reduce_sum(prims_surfaces, axis=1)
    
    surface_area *= 0.5
    
    return surface_area


@tf.function
@tf.custom_gradient
def wL_fn_EPO(node_bounds, node_min, node_max, beta, axis_points, parent_mask, parent_min, parent_max, primitive_cloud):

    prims_isect_node_mask, prims_in_sibling_node_mask, prims_outside_sibling_intersecting_node_inside_points_mask = get_prims_intersecting_node_mask(\
                                                                                                            node_bounds, parent_mask, primitive_cloud)
    
    intersecting_prims = tf.einsum('bij, bik -> bij', primitive_cloud, prims_isect_node_mask)
    surface_intersecting_prims = surface_prims_EPO(intersecting_prims)

    @tf.function
    def next_step_left_child():
        axis_points_in_sibling = tf.einsum('bij, bik -> bij', axis_points, prims_in_sibling_node_mask)
        # only keeps axis values of primitive points which acutally lay inside the bounds -> outside points are set to 0
        axis_points_out_sib_iscet_node_inside_points = axis_points * prims_outside_sibling_intersecting_node_inside_points_mask

        max_axis_point = tf.reduce_max(axis_points)

        mins_from_sibling = tf.reduce_min(axis_points_in_sibling, axis=2, keepdims=True)

        # needed to eliminate 0's for min reduction
        inverse_prims_outside_sibling_intersecting_node_inside_points_mask = (prims_outside_sibling_intersecting_node_inside_points_mask - 1) * (-1)
        axis_points_out_sib_iscet_node_inside_points += inverse_prims_outside_sibling_intersecting_node_inside_points_mask * max_axis_point

        temp_mins_from_out_sibling = tf.reduce_min(axis_points_out_sib_iscet_node_inside_points, axis=2, keepdims=True)
        prims_outside_sibling_intersecting_node_mask = tf.reduce_sum(prims_outside_sibling_intersecting_node_inside_points_mask, axis=-1, keepdims=True)
        prims_outside_sibling_intersecting_node_mask = tf.cast(prims_outside_sibling_intersecting_node_mask >= 1, tf.float32)
        mins_from_out_sibling = tf.einsum('bij, bik -> bij', temp_mins_from_out_sibling, prims_outside_sibling_intersecting_node_mask)

        # mins are all minimum (in split dimension) points from primitives which intersects the current node's bounds
        mins = mins_from_sibling + mins_from_out_sibling

        surface_reduction_offsets = tf.reduce_max(mins, axis=1)
        surface_reduction_offsets_prims_mask = tf.cast(tf.equal(surface_reduction_offsets[:, tf.newaxis, :], mins), tf.float32)

        difference_quotient_numerator = surface_prims_EPO(primitive_cloud * surface_reduction_offsets_prims_mask)
        difference_quotient_denominator = node_max - surface_reduction_offsets

        return difference_quotient_numerator, difference_quotient_denominator

    @tf.function
    def next_step_right_child():
        axis_points_in_sibling = tf.einsum('bij, bik -> bij', axis_points, prims_in_sibling_node_mask)
        # only keeps axis values of primitive points which acutally lay inside the bounds -> outside points are set to 0
        axis_points_out_sib_iscet_node_inside_points = axis_points * prims_outside_sibling_intersecting_node_inside_points_mask

        max_axis_point = tf.reduce_max(axis_points)        
        
        maxs_from_sibling = tf.reduce_max(axis_points_in_sibling, axis=2, keepdims=True)
        maxs_from_out_sibling = tf.reduce_max(axis_points_out_sib_iscet_node_inside_points, axis=2, keepdims=True)
        maxs = maxs_from_sibling + maxs_from_out_sibling

        maxs_transformed_for_min_reduction = (prims_isect_node_mask - 1) * (-max_axis_point)
        maxs_transformed_for_min_reduction = maxs_transformed_for_min_reduction + maxs
        surface_reduction_offsets = tf.reduce_min(maxs_transformed_for_min_reduction, axis=1)
        surface_reduction_offsets_prims_mask = tf.cast(tf.equal(surface_reduction_offsets[:, tf.newaxis, :], maxs), tf.float32)

        # swap of f(b) = (surface_intersecting_prims - surface_reduction) with f(a) = surface_intersecting_prims intended! 
        # -> swap direction of gradient because gradient would point to local minimum otherwise! (difference quotient = (f(b) - f(a)) / (b - a) )
        difference_quotient_numerator = surface_prims_EPO(primitive_cloud * surface_reduction_offsets_prims_mask)
        difference_quotient_denominator = surface_reduction_offsets - node_min

        return difference_quotient_numerator, difference_quotient_denominator
    
    @tf.function
    def grad(upstream):
        """ Custom gradient: Defined by the surface of the primitive(s) which are no longer intersecting the node when 
            reducing the node's AABB extent by moving the newly calculated bound inwards. """
        is_left_child = (node_min <= parent_min)[0]
        difference_quotient_numerator, difference_quotient_denominator = tf.cond(is_left_child, next_step_left_child, next_step_right_child)
        slope = tf.math.divide_no_nan(difference_quotient_numerator, difference_quotient_denominator)
        stepGrad = tf.clip_by_value(slope, 0.0, 1.0 / 0.0001)

        upstream_grad = stepGrad
        upstream_grad = tf.einsum('bi, bi -> bi', upstream, upstream_grad)
        # adapt here:
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(node_min >= parent_min, tf.float32))
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(node_max <= parent_max, tf.float32))

        upstream_grad_max = upstream_grad * 0
        upstream_grad_min = upstream_grad * 0
        if is_left_child:
            # return for node_max
            upstream_grad_max = upstream_grad
        else:
            # return for node_min
            upstream_grad_min = upstream_grad
        return None, upstream_grad_min, upstream_grad_max, None, None, None, None, None, None

    # 0.5 -> approximation of primitives surfaces which acutally lay inside the volume
    return 0.5 * (surface_intersecting_prims / surface_prims_EPO(primitive_cloud)), grad