from utils import *
from gcn import *
from sklearn.model_selection import train_test_split


def get_train_val_test(labels, seed=None, train_size=0.5, val_size=0.5):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    idx_train, idx_val = train_test_split(idx, random_state=None, train_size=train_size, test_size=val_size,
                                          stratify=labels)

    return idx_train, idx_val


if __name__ == "__main__":
    adj, features, labels = load_ndata(['../data/adj.pkl', '../data/feature.npy', '../data/train.npy'])
    # labels = labels - 1  # minus one for label index
    labels[labels==1] = labels[labels==1] - 1
    labels[labels>1] = labels[labels>1] - 2
    idx_train, idx_val = get_train_val_test(labels)

    model = MyModel("weights4layernorm")

    model.fit(features, adj, labels, idx_train, idx_val, train_iters=4000, verbose=True, patience=50)
