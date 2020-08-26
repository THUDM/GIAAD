from typing import Iterable, List, Union

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch


def split(
        labels: np.ndarray,
        train_size: float = 0.9,
        val_size: float = 0.1,
        random_state: int = None
) -> List[Union[np.ndarray, sp.spmatrix]]:
    """Return indices for splitting the labels into random train and validation subsets.

    Parameters
    ----------
    labels
        array of labels
    train_size
        Proportion of the dataset included in the train split.
    val_size
        Proportion of the dataset included in the validation split.
    random_state
        Random_state is the seed used by the random number generator;

    Returns
    -------
    list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    idx = np.arange(labels.shape[0])
    idx_train, idx_val = train_test_split(idx,
                                          random_state=random_state,
                                          train_size=train_size,
                                          test_size=val_size,
                                          stratify=labels)

    return idx_train, idx_val


def calc_accuracy(logits: torch.Tensor, labels: torch.Tensor, idx_test: np.ndarray) -> float:
    """
    Calculates the accuracy.

    Parameters
    ----------
    logits: torch.tensor
        The predicted logits.
    labels: torch.tensor
        The labels vector.
    idx_test: torch.tensor
        The indices of the test nodes.
    """
    accuracy = ((torch.argmax(logits, dim=-1) + 1)[idx_test] == labels[idx_test]).float().mean()
    return accuracy
