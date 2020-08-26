import scipy.sparse as sp
import numpy as np
import pickle
import sys
argv=sys.argv

with open(argv[1], "rb+") as f:
    adj = pickle.load(f)
    adj1=adj[659574:]
    print(adj1.shape)
    print(adj1.getnnz(axis=1).max())
    with open("submit/adj.pkl", "wb+") as f:
        pickle.dump(adj1, f)
feature=np.load(argv[2])
feature1=feature
print(feature1.shape)
print(np.max(feature),np.min(feature))
np.save("submit/feature.npy", feature1)

with open("submit/adj.pkl","rb+") as f:
    adj = pickle.load(f)
    print(adj1.shape)
    print(adj1.getnnz(axis=1).max())
feature=np.load("submit/feature.npy")
print(feature.shape)
# feature=np.load("feature.npy")
# print(np.max(feature),np.min(feature))