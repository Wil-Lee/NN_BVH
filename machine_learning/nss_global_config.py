# credits: https://github.com/cgaueb/nss
import os
import nss_tree_modules
import nss_loss
import nss_kd_tree

TREE_DATASET_DIR = os.path.join(os.getcwd(), 'machine_learning', 'datasets')
TRAIN_SCENES_DIR = os.path.join(os.getcwd(), 'machine_learning', 'train_scenes')
TEST_SCENES_DIR = os.path.join(os.getcwd(), 'machine_learning', 'test_scenes')
TREE_DIR =  os.path.join(os.getcwd(), 'machine_learning', 'trees')

# adjustable hyperparameters
pc_size = 2048
lvls = 4
epochs = 1
batch_amount = 2350 # after 2000 net seems to learn nothing new
test_primitive_clouds_per_scene = 64
EPO_SAH_alpha = 0.71
capacity = 128
batch_size = 64

i_isect = 1.0
""" Corresponds to C_tri of EPO paper. """
t_isect = 1.2
""" Corresponds to C_inn of EPO paper. """
t = 1.0
learning_rate = 0.0001
gamma = 1.0
beta = 1.0
# not supported for EPO:
train_unbalanced = False
sah_frag_name = '_fragments_{0}_sah'.format(pc_size)
vh_frag_name = '_fragments_{0}_vh'.format(pc_size)

init_config = {
    'point_cloud_size' : pc_size,
    'intersection_cost' : i_isect,
    'traversal_cost' : t_isect,
    't' : t,
    'gamma' : gamma,
    'layer_gamma' : 4.0,
    'beta' : beta,
    'penalty_fn' : nss_loss.penalty_tree_loss(slope=1.0),
    'loss_fn' : nss_loss.unsupervised_tree_loss(),
    'p_fn' : nss_tree_modules.p_eval(t_isect),
    'q_fn' : nss_tree_modules.q_eval(i_isect, beta),
    'greedy_q_fn' : nss_tree_modules.gr_q_eval(i_isect),
    'norm_factor' : 1.0 / (pc_size * i_isect),
    'tree_levels' : lvls,
    'dense_units_point_enc' : capacity,
    'dense_units_regr' : capacity,
    'learning_rate' : learning_rate,
    'train_unbalanced' : train_unbalanced,
    'checkpoint_window' : 15, ##################################### <-------- checkpoint
    'epochs' : epochs,
    'batch_size' : batch_size, 
    'train_scenes_dir' : TRAIN_SCENES_DIR,
    'test_scenes_dir' : TEST_SCENES_DIR,
    'output_tree_dir' : TREE_DIR,
    'batch_amount' : batch_amount,
    'test_sets' : test_primitive_clouds_per_scene,
    'EPO' : False
    }

def buildNetworkName(strat, lvls, pc_size, capacity) :
    return '{0}_kdtree_{1}lvl_{2}pc_{3}capacity'.format(
        'sah' if strat == nss_kd_tree.strategy.SURFACE_HEURISTIC_GREEDY else 'vh',
        str(lvls),
        str(pc_size),
        str(capacity),)

def build_network_name_EPO(lvls, pc_size, capacity, learning_rate, alpha, data_points):
    return'{0}lvl_{1}pc_{2}capacity_{3}learningRate_{4}_alpha_{5}dataPoints'.format(
        str(lvls), str(pc_size), str(capacity), str(learning_rate, str(alpha), str(data_points))
    )

epo_config = init_config.copy()
epo_config['tree_strat'] = nss_kd_tree.strategy.SURFACE_HEURISTIC_GREEDY
epo_config['name'] = build_network_name_EPO(
    lvls=epo_config['tree_levels'],
    pc_size=epo_config['point_cloud_size'],
    capacity=epo_config['dense_units_point_enc'],
    learning_rate=epo_config['learning_rate'],
    alpha=EPO_SAH_alpha,
    data_points=batch_amount * batch_size
)
epo_config['EPO'] = True
epo_config['weight_fn'] = nss_tree_modules.sah_eval() # TODO: maybe change
epo_config['pooling_fn'] = nss_tree_modules.pool_treelet_EPO(t, t_isect, 4 if init_config['train_unbalanced'] else 3,
    init_config['norm_factor'],
    beta,
    i_isect,
    EPO_SAH_alpha)

sah_config = init_config.copy()

sah_config['tree_strat'] = nss_kd_tree.strategy.SURFACE_HEURISTIC_GREEDY

sah_config['name'] = buildNetworkName(
    sah_config['tree_strat'], sah_config['tree_levels'],
    sah_config['point_cloud_size'],
    sah_config['dense_units_point_enc'])

sah_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + sah_frag_name)
#sah_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train_fragments_2048_masked_sah')
sah_config['test_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + sah_frag_name)
sah_config['valid_dir'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name),

sah_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + sah_frag_name + '.csv')
#sah_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train_fragments_2048_masked_sah.csv')
sah_config['test_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + sah_frag_name + '.csv')
sah_config['valid_csv'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name + '.csv')

sah_config['weight_fn'] = nss_tree_modules.sah_eval()
sah_config['pooling_fn'] = nss_tree_modules.pool_treelet(t, 4 if init_config['train_unbalanced'] else 3,
    nss_tree_modules.p_eval(t_isect),
    nss_tree_modules.q_eval(i_isect, beta),
    nss_tree_modules.gr_q_eval(i_isect),
    nss_tree_modules.sah_eval(),
    init_config['norm_factor'])
    
vh_config = init_config.copy()

vh_config['tree_strat'] = nss_kd_tree.strategy.VOLUME_HEURISTIC_GREEDY

vh_config['name'] = buildNetworkName(
    vh_config['tree_strat'], vh_config['tree_levels'],
    vh_config['point_cloud_size'],
    vh_config['dense_units_point_enc'])

vh_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + vh_frag_name)
vh_config['test_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + vh_frag_name)
vh_config['valid_dir'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name),

vh_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + vh_frag_name + '.csv')
vh_config['test_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + vh_frag_name + '.csv')
vh_config['valid_csv'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name + '.csv')

vh_config['weight_fn'] = nss_tree_modules.vh_eval()
vh_config['pooling_fn'] = nss_tree_modules.pool_treelet(t, 4 if init_config['train_unbalanced'] else 3,
    nss_tree_modules.p_eval(t_isect),
    nss_tree_modules.q_eval(i_isect, init_config['beta']),
    nss_tree_modules.gr_q_eval(i_isect),
    nss_tree_modules.vh_eval(),
    init_config['norm_factor'])
