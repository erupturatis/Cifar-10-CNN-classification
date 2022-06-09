import torch
import numpy as np
import torch.functional as F
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

def load_data():

    def unpickle():
        import pickle
        with open('data_batch_1', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    a = unpickle()

    data = a[b'data']
    labels =  np.array(a[b'labels'])
    batch_label = np.array(a[b'batch_label'])
    return data,labels


def visualize_image(data, idata:int = -1):
    if idata != -1:
        image = data[idata]
        image = image.reshape(3,32,32)
        print(image.shape)
        plt.imshow(image.T)
        plt.show()
    else:
        if type(data) == np.ndarray:
            d_type=type(data)
            raise Exception(f"wrong format not numpy array instead {d_type}")

        if len(data) != 3072:
            raise("wrong format incorrect length")

        image = data
        image = image.reshape(3,32,32)
        print(image.shape)
        plt.imshow(image.T)
        plt.show()
     
            

def create_model():
    pass


def train_model():
    pass


def data_loading(data,labels):

    dataT = torch.tensor(data)
    labelsT = torch.tensor(labels)

    train_data,train_labels,test_data,test_labels = train_test_split(dataT,labelsT,test_size=.1)
    train_data = TensorDataset(train_data,train_labels)
    test_data = TensorDataset(test_data,test_labels)
    
    batchsize = 10
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])




def main():
    data,labels = load_data()
 
    visualize_image(data[12])


if __name__=="__main__":
    main()