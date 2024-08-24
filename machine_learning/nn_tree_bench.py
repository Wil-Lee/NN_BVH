import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import math
import nn_AABB
import nn_BVH
import nn_loss
import nn_parser
import nn_types
import nss_data_stream
import nss_global_config
import nss_kd_tree
import nss_treeNet_model
import numpy as np
import os
import time

from contextlib import contextmanager
from tensorflow.keras import backend
from pathlib import Path

@contextmanager
def bench(name):
    print(f"\nStart {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"... {name} done in: " + str(end_time - start_time) + " seconds.")

class Model:
    def __init__(self, config):
        self.net = nss_treeNet_model.neural_kdtree(config, 'EPO_tree')
        self.net.load_trained_model(load_optimizer=False)

    def get_prediction(self, config, scene: nss_data_stream.scene, greedyInference = False):
        tree_structure, _ = self.net.predict_tree_EPO(scene)
        tree_structure = tree_structure[0]

        if not greedyInference :
            tree_structure = nss_kd_tree.kd_tree.preOrder_to_lvlOrder(config['tree_levels'], tree_structure)

        return tree_structure


def build_tree_from_nn_prediction(root_node: nn_BVH.BVHNode, tree_structure: np.ndarray):
    """ Builds BVH from prediciton. Returns true if all split offsets led to a valid split, else false. """
    axes = []
    offsets = []
    for level in tree_structure:
        if np.array_equal(level[:3], [1,0,0]):
            axes.append(nn_types.Axis.x)
        elif np.array_equal(level[:3], [0,1,0]):
            axes.append(nn_types.Axis.y)
        elif np.array_equal(level[:3], [0,0,1]):
            axes.append(nn_types.Axis.z)
        else:
            raise ValueError("Invalid split axis.")
        offsets.append(level[3])

    levels = int(math.log2(len(tree_structure) + 1))
    
    hierachy: list[nn_BVH.BVHNode] = [-1 for i in range(2 ** (levels + 1))]
    hierachy[0] = root_node
    hierachy_index = 1
    skip_index = []
    are_splits_valid = True
    
    for index in range(len(tree_structure)):
        if index not in skip_index:
            if hierachy[index].split(axes[index], offsets[index]):
                hierachy[hierachy_index] = (hierachy[index].left_child)
                hierachy[hierachy_index + 1] = (hierachy[index].right_child)
            else:
                skip_index.append(hierachy_index)
                skip_index.append(hierachy_index + 1)
        hierachy_index += 2

    for index, node_opt in enumerate(hierachy):
        if node_opt == -1:
            continue
        if len(node_opt.primitives) == 0:
            node_opt.parent.left_child = None
            node_opt.parent.right_child = None
            node_opt.parent.is_leaf = True
            are_splits_valid = False
        elif index >= (2 ** levels) - 1:
            node_opt.is_leaf = True
    if not are_splits_valid:
        print("Invalid predicted offsets!")


def main():
    print(tf.__version__)
    backend.clear_session()
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit('autoclustering')
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    ### remove later -> get bvh root node from scene #####################
    path = "machine_learning/bedroom_LowPoly.obj"
    p_mesh = nn_parser.parse_obj_file_with_meshes(path)
    p_mesh.primitives = nn_parser.scale_scene(p_mesh.primitives)
    aabb = nn_AABB.get_AABB_from_primitives(p_mesh.primitives)
    root_node_optimal = nn_BVH.BVHNode(aabb, p_mesh.primitives)
    root_node_nn_prediction = nn_BVH.BVHNode(aabb, p_mesh.primitives)
    ######################################################################
    
    config = nss_global_config.epo_config.copy()
    scenes = nss_data_stream.primitive_cloud_generator(config).get_base_scenes()

    alpha = nss_global_config.EPO_SAH_alpha
    levels = nss_global_config.lvls

    if 1:
        nn = Model(config)
        
        tree_structure = nn.get_prediction(config=config, scene=scenes[0])
        build_tree_from_nn_prediction(root_node_nn_prediction, tree_structure)
        sah_tree: float = nn_loss.SAH(root_node_nn_prediction)
        epo_tree: float = nn_loss.EPO(root_node_nn_prediction)

        print(f"pre_SAH: {sah_tree}")
        print(f"pre_EPO: {epo_tree}\n")

    with bench("Building tree"):
        nn_BVH.build_greedy_SAH_EPO_tree_single_thread(root_node_optimal, alpha, levels - 1, use_epo=True)

    sah_tree: float = nn_loss.SAH(root_node_optimal)
    epo_tree: float = nn_loss.EPO(root_node_optimal)

    print(f"SAH: {sah_tree}")
    print(f"EPO: {epo_tree}\n")
    root_node_optimal.print_tree()

if __name__ == "__main__":
    main()
