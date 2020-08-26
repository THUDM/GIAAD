import pickle
import numpy as np
import torch



import sys
import time


from dgl import DGLGraph
from dgl.transform import add_self_loop
def predict(adj,features):
    #adj_mat = pickle.load(open('adj_matrix_formal_stage.pkl', 'rb'))
    #feats = np.load('feature_formal_stage.npy')
    #true_labels = np.load('train_labels_formal_stage.npy') - 2
    #true_labels[ np.where(true_labels<0)] = 0
    
    t1 = time.time()
    #adj_mat = pickle.load(open(sys.argv[1],'rb'))
    #feats = np.load(sys.argv[2])
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dev = torch.device('cpu')
    features = torch.FloatTensor(features).to(dev)
    features = features / (features.norm(dim=1)[:, None] + 1e-8)
    
    graph = DGLGraph(adj)
    graph = add_self_loop(graph)
    
    
    model_filename = 'dminer/def_pool2.pt'

    model = torch.load(model_filename,map_location=dev) 
    
    model.eval()
    
    with torch.no_grad():
        logits = model(graph, features)
        #logits = logits[mask]
        _, labels = torch.max(logits, dim=1)
        
    labels = labels.cpu().detach().numpy()
    labels[ np.where(labels==0)] = -1
    labels = labels + 2
    
    return labels
    #print(labels)
    
    
    
