from gcn_t import *
from train_pyg import Pyg_Net, Pyg_GIN, Pyg_SAGE
import pickle as pkl
from utils import *
import numpy as np
import sys
import os
from torch.autograd import Variable
import time

def predict(adj,features):
    from collections import OrderedDict
    #dims=[100,64,19]
    #print(dims)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    processed_adj=GCNadj(adj)

    featuretensor=torch.FloatTensor(features)
    featuretensor=Variable(featuretensor,requires_grad=True)#.cuda()

    #print(processed_adj.row)
    sparserow=torch.LongTensor(processed_adj.row)#.unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col)#.unsqueeze(1)
    #sparseconcat=torch.cat((sparserow,sparsecol),0).cuda()
    sparseconcat=torch.stack([sparserow,sparsecol], dim=0)#.cuda()
    #print(sparseconcat)
    #sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    #sparsedata=torch.FloatTensor(processed_adj.data).cuda()
    #print(sparseconcat,sparsedata,processed_adj.shape)
    #adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()
    #print("here5")
    wd=0

    model=Pyg_SAGE()
    # model2=Pyg_GIN()
        
    best_val=0

    featuretensor.data[featuretensor.data.abs().ge(0.39).sum(1) > 20] = 0
    device = torch.device('cpu')
    mdl=torch.load("tsail/pyg_res_clamp04_sage_3layer_test_stage2.pth",map_location=device)
    def genkey(k):
        if "lin_rel" in k:
            return k.replace("lin_rel","lin_l")
        return k.replace("lin_root","lin_r")
        
    md=OrderedDict((genkey(k) if ('lin_r' in k ) else k, v) for k, v in mdl.items())
    model.load_state_dict(md)


    featuretensor = featuretensor.to(device)
    sparseconcat = sparseconcat.to(device)
    model=model.to(device)

    # model2.load_state_dict(torch.load("pyg_res_clamp04_GIN_stage2.pth", map_location={'cuda:2':'cpu'}))
    # model2=model2.cpu()

    testout=model(featuretensor, sparseconcat)
    # testout+=model2(featuretensor, sparseconcat)
    # testout*=0.5

    testoc=testout.argmax(1)

    testo=testoc.data.cpu().numpy()
    testo=testo+1
    #testo = testo.cpu()
    return testo

    
