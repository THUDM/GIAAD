from typing import Any, Dict, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train(model, features, edge_index, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, edge_weight=None, display_step=50):
    """Train a model using either standard or adversarial training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    features: torch.Tensor [n, d]
        Dense attribute matrix.
    edge_index: torch.Tensor [2, m]
        Edge indices.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    edge_weight: torch.Tensor [m]
        Edge weights.
    display_step : int
        How often to print information.
    Returns
    -------
    trace_val: list
        A list of values of the validation loss during training.
    """
    trace_train = []
    trace_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    for it in tqdm(range(max_epochs), desc='Training...'):
        logits = model(features, edge_index, edge_weight)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train] - 1)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        trace_train.append(loss_train.detach().item())

        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index, edge_weight)
            loss_val = F.cross_entropy(logits[idx_val], labels[idx_val] - 1)
            trace_val.append(loss_val.detach().item())
        model.train()

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            print(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_val, trace_train
