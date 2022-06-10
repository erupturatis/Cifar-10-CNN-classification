from cProfile import label
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

def load_initial_data():

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
     
            

def create_model(toggle:bool = False):
    class ConvolutionalNeuralNetwork(nn.Module):
        def __init__(self,print_toggle:bool = False) -> None:
            super().__init__()
            #3 * 32 * 32
            self.conv1 = nn.Conv2d(3,10,kernel_size=5,stride=1,padding=2) 
            # 32 + 2*2 - 4 = 32/2= 16 maxpool
            self.conv2 = nn.Conv2d(10,20,kernel_size=3,stride=1,padding=1)
            # 16 + 2*1 - 2 = 16/2 = 8 maxppol
            self.conv3 = nn.Conv2d(20,30,kernel_size=3,stride=1,padding=1)
            # 8 + 2*1 - 2 = 8/2 = 4 maxppol
            self.expected_size = 30*(4**2)
            self.fc1 = nn.Linear(self.expected_size,50)
            self.out = nn.Linear(50,10)

            self.print = print_toggle

        def forward(self,x):
            if self.print : print(x.shape)
            x = self.conv1(x)
            if self.print : print(x.shape)
            x = F.max_pool2d(x,2)
            x = F.relu(x)
            if self.print : print(x.shape)

            x = self.conv2(x)
            if self.print : print(x.shape)
            x = F.max_pool2d(x,2)
            x = F.relu(x)
            if self.print : print(x.shape)

            x = self.conv3(x)
            if self.print : print(x.shape)
            x = F.max_pool2d(x,2)
            x = F.relu(x)
            if self.print : print(x.shape)
            x = x.reshape((x.shape[0],self.expected_size))
            x = self.fc1(x)
            x = F.relu(x)

            x = self.out(x)
            return x

    net = ConvolutionalNeuralNetwork(toggle)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)

    return net,lossfun,optimizer



def train_model(net, lossfunction, optimizer, train_loader, device, epochs:int = 20):

    train_accuracy = []
    losses = []

    for epochi in range(epochs):
        print(epochi)
        batch_loss = []
        batch_accuracy = []
        i = 0
        for X,y in train_loader:
            i+=1
            if i%50 == 0 :print(f'epoch {epochi} and batch {i}')

            X = X.to(device)
            y = y.to(device)

            yHat = net(X)
            loss = lossfunction(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            matches = torch.argmax(yHat,axis=1) == y
            matchesNumeric = matches.float()
            accuracyPct = 100*torch.mean(matchesNumeric)
            batch_accuracy.append(accuracyPct)

        batch_accuracy = torch.tensor(batch_accuracy)
        batch_accuracy = batch_accuracy.to('cpu')
        batch_loss = torch.tensor(batch_loss)
        batch_loss = batch_loss.to('cpu')

        #print(batch_accuracy.device)
    
        train_accuracy.append(torch.mean(batch_accuracy))
        losses.append (torch.mean(batch_loss))
    
    return train_accuracy,losses

def data_processing(data,labels):

    dataT = torch.tensor(data).float()
    labelsT = torch.tensor(labels).long()

    dataT = dataT.reshape((dataT.shape[0],3,32,32))

    train_data,test_data,train_labels,test_labels = train_test_split(dataT,labelsT,test_size=.1)


    train_data = TensorDataset(train_data,train_labels)
    test_data = TensorDataset(test_data,test_labels)
    
    batchsize = 32
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader,test_loader

def plot_lists(*args):
    a = (len(args))
    print(a)
    fig,ax = plt.subplots(1,a)

    #plt.plot(*args)
    i = 0
    for list in args:
        ax[i].plot(list,'o')
        i+=1

    plt.show()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    data,labels = load_initial_data()
    train_loader, test_loader = data_processing(data,labels)
    net,lossfun,optimizer = create_model(False)
    net.to(device)
    epochs = 50
    train_acc,losses = train_model(net,lossfunction=lossfun,optimizer=optimizer,train_loader=train_loader,device=device,epochs=epochs)
    plot_lists(train_acc,losses)

    # visualize_image()


if __name__=="__main__":
    main()