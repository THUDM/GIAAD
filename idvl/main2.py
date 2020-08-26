from gcn_idvl import *
import pickle as pkl
from utils import *
import numpy as np
import sys
import os
from torch.autograd import Variable
import time

def predict(adj,features):
    # start = time.time()
    dims = [100, 256, 128, 64, 18]
    # print(dims)
   # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

   # adj = pkl.load(open(sys.argv[1], 'rb'))
   # features = np.load(sys.argv[2])
    processed_adj = GCNadj(adj)

    featuretensor = torch.FloatTensor(features).cuda()
    adjtensor = sparse_matrix_to_sparse_tensor(processed_adj)

    model = GCN_norm(len(dims) - 1, dims)

    model.load_state_dict(torch.load("idvl/weights"))
    model = model.cuda()
    model.eval()
    testout = model(featuretensor, adjtensor, dropout=0)

    testoc = testout.argmax(1)

    testo = testoc.data.cpu().numpy()
    testo[testo>0] = testo[testo>0] + 2
    testo[testo==0] = testo[testo==0] + 1
    return testo
    # f2 = open(sys.argv[3], 'w')
    
    # print(f'Done in time {time.time() - start}')
