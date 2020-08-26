
import numpy as np
import pickle as pkl


    
def load_xdata(dataset_str,test=10000,valid=10000):
    names=['adj','features','labels']
    objects=[]
    for i in range(len(names)):
        with open("{}_{}.pkl".format(dataset_str,names[i]),'rb') as f:
            objects.append(pkl.load(f))
    adj,features,labels=tuple(objects)

    processed_label=np.argmax(labels,axis=1)
    return adj, features,processed_label

def load_ndata(dataset_str):
    names=['adj','features','train']
    objects=[]
    for i in range(len(names)):
        with open("{}_{}.pkl".format(dataset_str,names[i]),'rb') as f:
            objects.append(pkl.load(f))
    adj,features,labels=tuple(objects)

    #processed_label=np.argmax(labels,axis=1)
    return adj, features,labels

def loadaf(dataset_str):
    names=['adj','features']
    print("here")
    objects=[]
    for i in range(len(names)):
        with open("{}_{}.pkl".format(dataset_str,names[i]),'rb') as f:
            objects.append(pkl.load(f))
    adj,features=tuple(objects)
    
    print("here2")
    #processed_label=np.argmax(labels,axis=1)
    return adj, features
