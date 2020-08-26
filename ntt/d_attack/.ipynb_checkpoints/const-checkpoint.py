ADJSIZE = 659574
TESTSIZE  = 50000
TRAINSIZE = ADJSIZE-TESTSIZE
FEATURE_DIM = 100
MAX_ADD_NODE = 500
MAX_ADD_EDGE = 100
SEED = 42

#Max:  1.6222137212753296 Min:  -1.735503077507019
FEATURE_MAX = 1.63
FEATURE_MIN = -1.74

ATTACK_VALUE = 99999999999

DATA_PATH = "../../mltrack2_data/kdd_cup_phase_two/"
#FILE_LIST = ["experimental_adj.pkl", "experimental_features.pkl", "experimental_train.pkl"]
FILE_LIST = ["adj_matrix_formal_stage.pkl", "feature_formal_stage.npy", "train_labels_formal_stage.npy"]
#MODEL_FNAME = "20200601_GCN_model.pkl"
#PARAMS_FNAME = "20200601_GCN_params.pkl"
