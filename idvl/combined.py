from gcn import *
import pickle as pkl
from utils import *
import numpy as np
import sys
import os
from torch.autograd import Variable
import time

if __name__ == "__main__":
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    adj = pkl.load(open(sys.argv[1], 'rb'))
    features = np.load(sys.argv[2])
    processed_adj = GCNadj(adj)

    featuretensor = torch.FloatTensor(features).cuda()
    adjtensor = sparse_matrix_to_sparse_tensor(processed_adj)

    model = MyModel("weights4layernorm")

    model.load_state_dict(torch.load("weights4layernorm"))
    model = model.cuda()
    model.eval()
    testout = model(featuretensor, adjtensor, dropout=0)

    testoc = testout.argmax(1)

    testo = testoc.data.cpu().numpy()
    testo[testo>0] = testo[testo>0] + 2
    testo[testo==0] = testo[testo==0] + 1
    # f2 = open(sys.argv[3], 'w')
    f2 = open(sys.argv[3], 'w')

    for i in range(len(testo)):
        print(int(testo[i]), file=f2)

    f2.close()
    print(f'Done in time {time.time() - start}')