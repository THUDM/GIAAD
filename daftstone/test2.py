import scipy.sparse as sp
import numpy as np
import pickle
with open("../temp/data/adj_matrix_formal_stage.pkl","rb+") as f:
    adj=pickle.load(f)
    print(adj.getnnz(axis=1).max())
with open("../temp/data/adj_matrix_formal_stage.pkl", "rb+") as f:
    adj = pickle.load(f)
    adj1 = adj.copy()
    print(adj1.shape)
    print(adj1.getnnz(axis=1).max())
    with open("submit/adj.pkl", "rb+") as f:
        adj2 = pickle.load(f)
        adj_temp = sp.vstack([adj1, adj2[:,:659574]])
        adj_temp2=sp.vstack([adj2[:,:659574].T,adj2[:,659574:]])
        adj=sp.hstack([adj_temp,adj_temp2]).tocsr()
        print(adj.shape)
        assert np.abs(adj - adj.T).sum() == 0, "symmetric"
        with open("adj.pkl", "wb+") as f:
            pickle.dump(adj, f)
feature = np.load("submit/feature.npy")
feature1=np.load("../temp/data/feature_formal_stage.npy")
feature1 = np.concatenate([feature1,feature],axis=0)
print(feature1.shape)
np.save("feature.npy", feature1)