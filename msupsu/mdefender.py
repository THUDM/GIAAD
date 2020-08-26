import numpy as np
import torch.nn.functional as F
import sys
from dpr import GCN
import argparse
import torch
import pickle as pkl
import os
import sys

def predict(adj,features):
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    nhid = [128, 128, 128]
    # nhid = [256, 256, 256]
    gcn = GCN(nfeat=features.shape[1],
          nhid=nhid,
          nclass=20,
          dropout=0.5, device=device,
          lr=0.01)

    adj = gcn.drop_dissimilar_edges(features, adj)
    weights = torch.load('msupsu/mlgcn_64000.pt', map_location=torch.device(device))
    gcn.load_state_dict(weights)

    gcn = gcn.to(device)
    gcn.eval()
    output = gcn.predict(features, adj)
    preds = output.argmax(1).cpu().numpy()
    return preds
    
