import numpy as np
import torch


def accuracy(output, labels):
    if type(output) == torch.Tensor:
        output_np = output.detach().cpu().numpy()
        output_np =  output_np.argmax(1)
    if type(labels) == torch.Tensor:
        labels_np = labels.detach().cpu().numpy()
    return sum(output_np==labels_np) / len(labels)


def load_ndata(names):
    objects = []
    for i in range(len(names)):
        objects.append(np.load(names[i], allow_pickle=True))
    return tuple(objects)


def sparse_matrix_to_sparse_tensor(processed_adj):
    sparserow = torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol = torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1).cuda()
    sparsedata = torch.FloatTensor(processed_adj.data).cuda()
    adjtensor = torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(processed_adj.shape)).cuda()
    return adjtensor