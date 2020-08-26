import pickle as pkl
import numpy as np
import sklearn
import torch

class Dataset:

    def __init__(self, tensor=False, device='cpu'):

        with open('kdd_cup_phase_two/adj_matrix_formal_stage.pkl', 'rb') as f:
            self.adj = pkl.load(f)

        self.features = np.load('kdd_cup_phase_two/feature_formal_stage.npy')
        self.train_labels = np.load('kdd_cup_phase_two/train_labels_formal_stage.npy')
        if tensor:
            self.features = torch.FloatTensor(self.features).to(device)
            self.train_labels = torch.LongTensor(self.train_labels).to(device)
