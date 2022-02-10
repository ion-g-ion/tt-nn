import torch as tn
from torchvision import datasets, transforms
import torchtt as tntt
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import datetime

data_dir = 'Cat_Dog_data/'

transform_train = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])
dataset_train = datasets.ImageFolder(data_dir, transform=transform_train)
dataloader_train = tn.utils.data.DataLoader(dataset_train, batch_size=3250, shuffle=True, pin_memory=True)

transform_test = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])
dataset_test = datasets.ImageFolder(data_dir, transform=transform_test)
dataloader_test = tn.utils.data.DataLoader(dataset_test, batch_size=3500, shuffle=True, pin_memory=True)


class BasicTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tntt.nn.LinearLayerTT([3,8,8,8,8], [8,4,4,4,4], [1,3,2,2,2,1])
        self.ttl2 = tntt.nn.LinearLayerTT([8,4,4,4,4], [4,2,2,2,2], [1,2,2,2,2,1])
        self.ttl3 = tntt.nn.LinearLayerTT([4,2,2,2,2], [2,2,2,2,2], [1,2,2,2,2,1])
        self.linear = nn.Linear(32, 2, dtype = tn.float32)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.ttl1(x)
        x = tn.relu(x)
        x = self.ttl2(x)
        x = tn.relu(x)
        x = self.ttl3(x)
        x = tn.relu(x)
        x = x.view(-1,32)
        x = self.linear(x)
        return self.logsoftmax(x)

device_name = 'cuda:0'
model = BasicTT()        
model.to(device_name)


optimizer = tn.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
# optimizer = tn.optim.Adam(model.parameters(), lr=0.001)
loss_function = tn.nn.CrossEntropyLoss()

def do_epoch(i):
    
    loss_total = 0.0
    
    for k, data in enumerate(dataloader_train):
        # tme = datetime.datetime.now()
        inputs, labels = data[0].to(device_name), data[1].to(device_name)
        # tme = datetime.datetime.now() - tme
        # print('t1',tme)
        
        # tme = datetime.datetime.now()
        inputs = tn.reshape(inputs,[-1,3,8,8,8,8])
        # tme = datetime.datetime.now() - tme
        # print('t2',tme)
        
        # tme = datetime.datetime.now()
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # tme = datetime.datetime.now() - tme
        # print('t3',tme)
        
        loss_total += loss.item()
        # print('\t\tbatch %d error %e'%(k+1,loss))
    return loss_total/len(dataloader_train)

def test_loss():
    loss_total = 0 
    for data in dataloader_test:
        inputs, labels = data[0].to(device_name), data[1].to(device_name)
        inputs = tn.reshape(inputs,[-1,3,8,8,8,8])
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss_total += loss.item()
        
    return loss_total/len(dataloader_test)
        
        
n_epochs = 1000

history_test = []
history_train = []
for epoch in range(n_epochs):
    print('Epoch ',epoch+1)
    time_epoch = datetime.datetime.now()
    
    model.train(True)
    average_loss = do_epoch(epoch)
    model.train(False)
    
    average_test_loss = test_loss()
    time_epoch = datetime.datetime.now() - time_epoch
    
    print('\tTraining loss %e test loss %e'%(average_loss,average_test_loss))
    print('\tTime for the epoch',time_epoch)
    history_test.append(average_test_loss)
    history_train.append(average_loss)
    
plt.figure()
plt.plot(np.arange(n_epochs)+1,np.array(history_train))
plt.plot(np.arange(n_epochs)+1,np.array(history_test))
plt.legend(['training','test'])