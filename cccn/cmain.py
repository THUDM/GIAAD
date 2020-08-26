import os
import sys

from gcn_cccn import *
import pickle as pkl
from utils import *
import numpy as np


from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


def predict(adj,features):
    
    device,use_cuda = get_cuda()


    dims=[100,128,20]
    #print(dims)

        
    # do some preocessing
    processed_adj=GCNadj(adj, features)
    features = StandardScaler().fit(features).transform(features)

    
    featuretensor=torch.FloatTensor(features)
    featuretensor=Variable(featuretensor,requires_grad=True)

    #print(processed_adj.row)
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1)
    sparsedata=torch.FloatTensor(processed_adj.data)
    #print(sparseconcat,sparsedata,processed_adj.shape)
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape))
    if use_cuda:
        sparseconcat=sparseconcat.cuda()
        sparsedata=sparsedata.cuda()
        adjtensor=adjtensor.cuda()
        featuretensor=featuretensor.cuda()


    nhid = dims[1]
    nfeat = dims[0]
    nclass = dims[2]
    model1=GCN(nfeat,nhid,nclass,device=device)
    model1.load_state_dict(torch.load("cccn/cccn0726-I+A-3000"))
    model=GCN_norm2(model1)

    #model=model.cpu()
    if use_cuda:
        model=model.cuda()
    else:
        model=model.cpu()
    testout=model(featuretensor,adjtensor)

    testoc=testout.argmax(1)

    testo=testoc.data.cpu().numpy()

    return testo
