import random
import Dataset
import GCN
import gcnutils as utils
import torch
import numpy as np
import torch.nn.functional as F
from multiprocessing import Process

data_path = {
    'adj': '/data1/qyf/data//final_new_adj.pkl',  # experimental_adj.pkl',
    'feat': '/data1/qyf/data//feature_formal_stage.npy',  # /experimental_features.pkl',
    'label': '/data1/qyf/data//train_labels_formal_stage.npy',  # experimental_train.pkl',
}

# 1 basic parameter config
seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("Read and processing data...")
data = Dataset.Dataset(data_path=data_path)


# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
adj, features, labels = data.adj, data.features, data.labels

adj = utils.normalize_adj(adj)
g_adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
g_features = utils.sparse_mx_to_torch_sparse_tensor(features)
g_labels = torch.LongTensor(labels)


def train_model(model_id, idx_train, idx_val, idx_test):
    '''
    @Description    :
    @Time    :2020/07/24 13:59:21
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    # data preprare
    use_cuda = True
    if use_cuda:
        device = torch.device(str.format('cuda:{}', model_id)
                              ) if torch.cuda.is_available() else 'cpu'
    else:
        device = torch.device('cpu')

    adj = g_adj.to(device)
    features = g_features.to(device)
    labels = g_labels.to(device)

    # 3 Setup victim model
    print("Creating GCN model...")
    victim_model = GCN.GCN(nfeat=features.shape[1], lr=0.001, nclass=int(labels.max().item())+1,
                           nhid=100, dropout=0.5, weight_decay=5e-4, device=device)

    victim_model = victim_model.to(device)
    print("Start fitting GCN...")
    victim_model.fit(features, adj, labels, idx_train, idx_val,
                     train_iters=100000, normalize=False, verbose=True)
    # setattr(victim_model, 'norm_tool',  GraphNormTool(normalize=True, gm='gcn', device=device))

    # 4 validation on 80% test data
    output = victim_model.predict(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = GCN.accuracy(output[idx_test], labels[idx_test])
    print("Init test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    torch.save(victim_model.state_dict(), 'gcn_%s.pth' % (str(model_id)))
    print(str.format("Finished Model :{}", i))


if __name__ == '__main__':
    k_ford_idx = data.k_ford_idx
    pss = []
    for i, idx in enumerate(k_ford_idx):
        i = i+1
        print(str.format("Train Model :{}", i))
        idx_train, idx_val, idx_test = idx
        p = Process(target=train_model, args=[i, idx_train, idx_val, idx_test])
        p.daemon = True
        p.start()
        pss.append(p)

    for p in pss:
        p.join()

    print("K ford Finished....")
