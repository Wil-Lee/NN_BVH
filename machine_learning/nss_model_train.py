# credits: https://github.com/cgaueb/nss
import os
import tensorflow as tf
tf.random.set_seed(23445)

from tensorflow.keras import backend
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import nss_global_config
import nss_treeNet_model


def train_sah() :
    net = nss_treeNet_model.neural_kdtree(nss_global_config.sah_config, 'test_tree')
    net.train()
    #net.continue_training()

def train_vh() :
    net = nss_treeNet_model.neural_kdtree(nss_global_config.vh_config, 'test_tree')
    net.train()
    #net.continue_training()

def train_EPO():
    net = nss_treeNet_model.neural_kdtree(nss_global_config.epo_config, 'EPO_tree')
    net.train_EPO()
    #net.continue_training()


def main():
    train_EPO()
    #train_sah()
    #train_vh()

if __name__ == "__main__":
    print(tf.__version__)
    backend.clear_session()
    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    cudnn_version = sys_details["cudnn_version"]  
    print('Cuda vs: {0} - Cudnn vs: {1}'.format(cuda_version, cudnn_version))
    
    tf.config.run_functions_eagerly(False)

    tf.config.optimizer.set_jit('autoclustering')
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    if not os.path.exists(os.path.join(os.getcwd(), 'machine_learning', 'metadata')) :
        os.mkdir(os.path.join(os.getcwd(), 'machine_learning', 'metadata'))

    if not os.path.exists(os.path.join(os.getcwd(), 'machine_learning', 'plots')) :
        os.mkdir(os.path.join(os.getcwd(), 'machine_learning', 'plots'))

    main()