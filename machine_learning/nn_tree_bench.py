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

    def get_prediction(self, config, scene: nss_data_stream.Scene, greedyInference = False):
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
    
    hierachy: list[nn_BVH.BVHNode] = [-1 for i in range(2 ** (levels + 1) - 1)]
    hierachy[0] = root_node
    hierachy_index = 1
    skip_index = []
    
    for index in range(len(tree_structure)):
        hierachy[index].split(axes[index], offsets[index])
        hierachy[hierachy_index] = hierachy[index].left_child
        hierachy[hierachy_index + 1] = hierachy[index].right_child
        hierachy_index += 2
        if len(hierachy[index].left_child.primitives) <= nn_BVH.MAX_PRIMITIVES_PER_LEAF:
            hierachy[index].left_child.is_leaf = True
        if len(hierachy[index].right_child.primitives) <= nn_BVH.MAX_PRIMITIVES_PER_LEAF:
            hierachy[index].right_child.is_leaf

    for i in range((2 ** levels) - 1, len(hierachy)):
            hierachy[i].is_leaf = True



def main():
    print(tf.__version__)
    backend.clear_session()
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit('autoclustering')
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    evaluate_model: bool = 1
    if evaluate_model:
        config = nss_global_config.epo_config.copy()
        nn = Model(config)
        generator = nss_data_stream.primitive_cloud_generator(config)
        scenes_for_prediction, scene_names_pred = generator.get_base_scenes_for_nn_prediction()
        scenes_for_evaluation = generator.get_base_scenes_for_evalutation()
        assert len(scenes_for_prediction) == len(scenes_for_evaluation), "amount of train and test scenes are not equal"
        bench_scenes = generator.load_bench_scenes()

        for i in range(len(scenes_for_prediction)):
            scene_name_pred = scene_names_pred[i]
            scene_eval = scenes_for_evaluation[scene_name_pred]
            scene_name_eval = scene_eval.name
            assert scene_name_eval == scene_name_pred, f"train: {scene_name_pred} and test {scene_name_eval} scene are not the same"

            tree_structure = nn.get_prediction(config=config, scene=scenes_for_prediction[i])

            scene_mesh = scene_eval.scene
            scene_aabb = nn_AABB.get_AABB_from_primitives(scene_mesh.primitives)
            root_node_nn_prediction = nn_BVH.BVHNode(scene_aabb, scene_mesh.primitives)

            build_tree_from_nn_prediction(root_node_nn_prediction, tree_structure)

            print(f"\n\nEvaluating SAH and EPO cost for: {scene_name_eval}...")
            sah_tree: float = nn_loss.SAH(root_node_nn_prediction)
            epo_tree: float = nn_loss.EPO(root_node_nn_prediction)

            print(f"{scene_name_eval} pre_SAH: {sah_tree}")
            print(f"{scene_name_eval} pre_EPO: {epo_tree}\n")
                root_node_nn_prediction.print_tree()
        
        
        for s in bench_scenes.values():
            scene_mesh = s.scene
            scene_name = s.name
            scene_prim_cloud = s.prim_cloud
            tree_structure = nn.get_prediction(config=config, scene=scene_prim_cloud)

            scene_aabb = nn_AABB.get_AABB_from_primitives(scene_mesh.primitives)
            root_node_nn_prediction = nn_BVH.BVHNode(scene_aabb, scene_mesh.primitives)

            build_tree_from_nn_prediction(root_node_nn_prediction, tree_structure)

            print(f"\n\nEvaluating SAH and EPO cost for: {scene_name}...")
            sah_tree: float = nn_loss.SAH(root_node_nn_prediction)
            epo_tree: float = nn_loss.EPO(root_node_nn_prediction)

            print(f"{scene_name} pre_SAH: {sah_tree}")
            print(f"{scene_name} pre_EPO: {epo_tree}\n")
            root_node_nn_prediction.print_tree()


    else:
        # manual selection
        path = "machine_learning/test_scenes/bedroom_LowPoly_test.obj"
    p_mesh = nn_parser.parse_obj_file_with_meshes(path)
        nn_parser.scale_scene(p_mesh.primitives)
    aabb = nn_AABB.get_AABB_from_primitives(p_mesh.primitives)
    root_node_optimal = nn_BVH.BVHNode(aabb, p_mesh.primitives)
    
        levels = nss_global_config.lvls
    alpha = nss_global_config.EPO_SAH_alpha
    with bench("Building tree"):
            nn_BVH.build_greedy_SAH_tree_tf(root_node_optimal, alpha, levels - 1, batch_size_gpu=2048, print_progress=True)#, max_workes=7, use_epo=False)

    sah_tree: float = nn_loss.SAH(root_node_optimal)
    epo_tree: float = nn_loss.EPO(root_node_optimal)

    print(f"SAH: {sah_tree}")
    print(f"EPO: {epo_tree}\n")
    root_node_optimal.print_tree()

if __name__ == "__main__":
    main()
