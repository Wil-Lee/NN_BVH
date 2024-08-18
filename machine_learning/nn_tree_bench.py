import nn_AABB
import nn_BVH
import nn_loss
import nn_parser
import nss_global_config

from contextlib import contextmanager
import time

@contextmanager
def bench(name):
    print(f"\nStart {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"... {name} done in: " + str(end_time - start_time) + " seconds.")

def main():
    path = "machine_learning/bedroom_LowPoly.obj"
    p_mesh = nn_parser.parse_obj_file_with_meshes(path)
    p_mesh.primitives = nn_parser.scale_scene(p_mesh.primitives)
    aabb = nn_AABB.get_AABB_from_primitives(p_mesh.primitives)
    root_node = nn_BVH.BVHNode(aabb, p_mesh.primitives)

    alpha = nss_global_config.EPO_SAH_alpha
    levels = nss_global_config.lvls

    with bench("Building tree"):
        nn_BVH.build_greedy_SAH_EPO_tree_multi_thread(root_node, alpha, levels, use_epo=True)

    sah_tree: float = nn_loss.SAH(root_node)
    epo_tree: float = nn_loss.EPO(root_node)

    print(f"SAH: {sah_tree}")
    print(f"EPO: {epo_tree}\n")
    root_node.print_tree()

if __name__ == "__main__":
    main()
