# credits: https://github.com/cgaueb/nss
import os
import nss_treeNet_model
import tensorflow as tf
from tensorflow.keras import backend
import nss_global_config

def train_sah() :
    net = nss_treeNet_model.neural_kdtree(nss_global_config.sah_config, 'test_tree')
    net.train()
    #net.continue_training()

def train_vh() :
    net = nss_treeNet_model.neural_kdtree(nss_global_config.vh_config, 'test_tree')
    net.train()
    #net.continue_training()

def train_EPO():
    net = nss_treeNet_model.neural_kdtree(nss_global_config.vh_config, 'EPO_tree')
    net.train()

    
def main():
    train_sah()
    #train_vh()

if __name__ == "__main__":
    print(tf.__version__)
    backend.clear_session()
    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    cudnn_version = sys_details["cudnn_version"]  
    print('Cuda vs: {0} - Cudnn vs: {1}'.format(cuda_version, cudnn_version))
    
    tf.config.run_functions_eagerly(False)
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit('autoclustering')
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    if not os.path.exists(os.path.join(os.getcwd(), 'metadata')) :
        os.mkdir(os.path.join(os.getcwd(), 'metadata'))

    if not os.path.exists(os.path.join(os.getcwd(), 'plots')) :
        os.mkdir(os.path.join(os.getcwd(), 'plots'))

    main()