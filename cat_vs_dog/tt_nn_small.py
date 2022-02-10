import torch as tn
from torchvision import datasets, transforms
import torchtt as tntt
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import datetime

device_name = 'cuda:0'
data_dir = 'Cat_Dog_data/'

transform_train = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]) #, transforms.Normalize(tn.tensor([0.4885, 0.4525, 0.4163]), tn.tensor([0.2549, 0.2476, 0.2495]))])
dataset_train = datasets.ImageFolder('train', transform=transform_train)
dataloader_train = tn.utils.data.DataLoader(dataset_train, batch_size=3250, shuffle=True, pin_memory=True, num_workers=0)

transform_test = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]) #, transforms.Normalize(tn.tensor([0.4885, 0.4525, 0.4163]), tn.tensor([0.2549, 0.2476, 0.2495])) ])
dataset_test = datasets.ImageFolder('test', transform=transform_test)
dataloader_test = tn.utils.data.DataLoader(dataset_test, batch_size=3500, shuffle=True, pin_memory=True, num_workers=0)

inputs_train = list(dataloader_train)[0][0].to(device_name)
labels_train = list(dataloader_train)[0][1].to(device_name)

inputs_test = list(dataloader_test)[0][0].to(device_name)
labels_test = list(dataloader_test)[0][1].to(device_name)

class BasicTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tntt.nn.LinearLayerTT([3,8,8,8,8], [8,10,10,10,10], [1,2,2,2,2,1])
        self.ttl2 = tntt.nn.LinearLayerTT([8,10,10,10,10], [8,4,4,4,4], [1,2,2,2,2,1])
        self.ttl3 = tntt.nn.LinearLayerTT([8,4,4,4,4], [3,3,3,3,3], [1,2,2,2,2,1])
        self.ttl4 = tntt.nn.LinearLayerTT([3,3,3,3,3], [3,3,3,3,3], [1,1,1,1,1,1])
        self.ttl5 = tntt.nn.LinearLayerTT([3,3,3,3,3], [3,3,3,3,3], [1,1,1,1,1,1])
        self.linear = nn.Linear(3**5, 2, dtype = tn.float32)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.ttl1(x)
        x = tn.relu(x)
        x = self.ttl2(x)
        x = tn.relu(x)
        x = self.ttl3(x)
        x = tn.relu(x)
        x = self.ttl4(x)
        x = tn.relu(x)
        x = self.ttl5(x)
        x = tn.relu(x)
        x = x.view(-1,2**5)
        x = self.logsoftmax(self.linear(x))
        return x


model = BasicTT()        
model.to(device_name)


optimizer = tn.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = tn.optim.Adam(model.parameters(), lr=0.001)
loss_function = tn.nn.CrossEntropyLoss()

def do_epoch(i):
    
    loss_total = 0.0
    
    # tme = datetime.datetime.now()
    # tme = datetime.datetime.now() - tme
    # print('t1',tme)
    
    # tme = datetime.datetime.now()
    inputs = tn.reshape(inputs_train,[-1,3,8,8,8,8])
    # tme = datetime.datetime.now() - tme
    # print('t2',tme)
    
    # tme = datetime.datetime.now()
    optimizer.zero_grad()
    # Make predictions for this batch
    outputs = model(inputs)
    # Compute the loss and its gradients
    loss = loss_function(outputs, labels_train)
    loss.backward()
    # Adjust learning weights
    optimizer.step()
    # tme = datetime.datetime.now() - tme
    # print('t3',tme)
    
    loss_total += loss.item()
    # print('\t\tbatch %d error %e'%(k+1,loss))
    return loss_total

def test_loss():
    inputs = tn.reshape(inputs_test,[-1,3,8,8,8,8])
    outputs = model(inputs)
    loss = loss_function(outputs, labels_test)
    loss_total = loss.item()
    
    return loss_total
       
def test_accuracy():
    inputs = tn.reshape(inputs_test,[-1,3,8,8,8,8])
    outputs = model(inputs)
    accuracy = tn.sum(tn.max(outputs,1)[1] == labels_test).cpu()/inputs.shape[0]   

    return accuracy
            
def train_accuracy():
    inputs = tn.reshape(inputs_train,[-1,3,8,8,8,8])
    outputs = model(inputs)
    accuracy = tn.sum(tn.max(outputs,1)[1] == labels_train).cpu()/inputs.shape[0]   

    return accuracy
        
n_epochs = 100000

history_test = []
history_train = []
for epoch in range(n_epochs):
    print('Epoch ',epoch+1)
    time_epoch = datetime.datetime.now()
    
    model.train(True)
    average_loss = do_epoch(epoch)
    model.train(False)
    
    accuracy = test_accuracy()
    time_epoch = datetime.datetime.now() - time_epoch
    
    accuracy_train = train_accuracy()
    
    print('\tTraining loss %e test accuracy %e training accuracy'%(average_loss,accuracy_train))
    print('\tTime for the epoch',time_epoch)
    history_test.append(accuracy)
    history_train.append(average_loss)
    
plt.figure()
plt.plot(np.arange(n_epochs)+1,np.array(history_train))
plt.plot(np.arange(n_epochs)+1,np.array(history_test))
plt.legend(['training','test'])