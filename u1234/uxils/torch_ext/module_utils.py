from torch import nn, optim
# from .loss import FocalLoss, soft_margin_focal_loss


OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'adamw': optim.AdamW,
}

ACTIVATIONS = {
    'tanh': nn.Tanh,
    'tanhshrink': nn.Tanhshrink,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
    'softshrink': nn.Softshrink,
    'softsign': nn.Softsign,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'elu': nn.ELU,
    'leakyrelu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'selu': nn.SELU,
}

NORMALIZATIONS = {
    'batch': nn.BatchNorm1d,
    'bn': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
}


CRITERIONS = {
    'ce': nn.CrossEntropyLoss,
    'mse': nn.MSELoss,
    'l2': nn.MSELoss,
    'l1': nn.L1Loss,
    'bce': nn.BCEWithLogitsLoss,
    # 'focal': FocalLoss,
    # 'soft_margin_focal': soft_margin_focal_loss,
}


def is_subclass(obj, classinfo):
    try:
        return issubclass(obj, classinfo)
    except Exception:
        pass
    return False


def init_activation(activation):
    if activation is None:
        return None
    if isinstance(activation, str) and activation.lower() in ACTIVATIONS:
        return ACTIVATIONS[activation.lower()]()

    if is_subclass(activation, nn.Module):
        return activation()

    raise ValueError('No such activation: "{}"'.format(activation))


def init_normalization(normalization):
    if normalization is None:
        return None
    if isinstance(normalization, str) and normalization in NORMALIZATIONS:
        return NORMALIZATIONS[normalization]
    if is_subclass(normalization, nn.Module):
        return normalization
    raise ValueError('No such normalization: "{}"'.format(normalization))


def init_optimizer(optimizer):
    if isinstance(optimizer, str) and optimizer.lower() in OPTIMIZERS:
        return OPTIMIZERS[optimizer.lower()]
    if is_subclass(optimizer, optim.Optimizer):
        return optimizer
    raise ValueError('No such optimizer: "{}"'.format(optimizer))


def init_criterion(criterion):
    if isinstance(criterion, str) and criterion.lower() in CRITERIONS:
        return CRITERIONS[criterion.lower()]()
    if is_subclass(criterion, nn.Module):
        return criterion
    raise ValueError(f'No such criterion: "{criterion}".\nAvailable options: {CRITERIONS.keys()}')


def available_activations():
    return list(ACTIVATIONS.keys())


def available_optimizers():
    return list(OPTIMIZERS.keys())
