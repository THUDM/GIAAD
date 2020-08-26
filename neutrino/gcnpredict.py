import torch
import numpy as np
import GCN
import gcnutils as utils
import datetime


def predict(adj, features, output_prob=False):

    start = datetime.datetime.now()
    use_cuda = True
    if use_cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    else:
        device = torch.device('cpu')

    # load the trained GCN to attack

    victim_model = GCN.GCN(nfeat=100, lr=0.01, nclass=20,
                           nhid=100, dropout=0.5, weight_decay=5e-4, device=device)
    victim_model.load_state_dict(torch.load(str.format('neutrino/gcn_{}.pth', 32)))
    victim_model = victim_model.to(device)

    stage1 = datetime.datetime.now()

    time_span = (stage1 - start).seconds
    print(str.format("--------------GCN [model To Devide]  : {}s", time_span))

    # normalize
    adj_norm = utils.normalize_adj(adj)
    adj_norm = utils.sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
    features = torch.FloatTensor(features).to(device)

    stage2 = datetime.datetime.now()
    time_span = (stage2 - stage1).seconds
    print(str.format("--------------GCN [Data To Devide] : {}s", time_span))

    output = victim_model.predict(features, adj_norm)

    stage3 = datetime.datetime.now()
    time_span = (stage3 - start).seconds
    print(str.format("--------------GCN [Model Predict]  : {}s", time_span))

    if output_prob == False:
        result = output.max(1)[1].cpu().detach().numpy()
    else:
        result = output.cpu().detach().numpy()

    stage4 = datetime.datetime.now()
    time_span = (stage4 - stage3).seconds
    print(str.format("--------------GCN [Result Collect]  : {}s", time_span))
    return result
