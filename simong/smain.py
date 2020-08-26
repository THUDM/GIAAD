import pickle
import numpy as np
import sys

import torch
from torch_geometric.utils import from_scipy_sparse_matrix

from cup.models import APPNPModel

MODEL_FILE = 'simong/model.pkl'
MODEL_CHECKPOINT = 'simong/model.checkpoint'

def predict(adj,features):

    
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index, edge_weight = edge_index.cuda(), edge_weight.float().cuda()
    features = torch.from_numpy(features).float().cuda()

    n, d = features.shape

    print('Data loaded')

    model = APPNPModel(
        n_features=d,
        n_classes=19,
        hidden_dimensions=[128],
        alpha=0.01,
        do_use_dropout_for_propagation=True
    ).cuda()
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))

    # with open(MODEL_FILE, 'rb') as fp:
    #     model = pickle.load(fp).cuda()
    model.eval()

    print('Model loaded')

    logits = model(features, edge_index, edge_weight)

    print('Prediction finished')

    test_out = (logits.argmax(-1) + 1).cpu().numpy()
    # test_out = test_out.cpu()
    # test_out = test_out.numpy()
    return test_out
