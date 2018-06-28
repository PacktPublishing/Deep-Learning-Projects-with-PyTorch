# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:08:31 2018

@author: Ashish Bhatia
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import transforms
import torchvision.datasets as datasets

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)

print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())

print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

# Make dataset iterable
batch_size = 100
n_iters = 3000

num_epochs = 5

train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(train_loader)
#Create Model Class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        #Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, 
                              out_channels=16,
                             kernel_size=5,
                             stride=1,
                             padding=2)
        self.relu1 = nn.ReLU()
        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, 
                              out_channels=32,
                             kernel_size=5,
                             stride=1,
                             padding=2)
        self.relu2 = nn.ReLU()
        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
                
        self.fc1 = nn.Linear(32*7*7, 10)
    
    def forward(self, x):
        # C1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        #Maxpool1
        out = self.maxpool1(out)
        
        #c1
        out = self.cnn2(out)
        out = self.relu2(out)
        
        #Maxpool1
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        #Linear Function
        out = self.fc1(out)
        
        return out


model = CNNModel()

criterion = nn.CrossEntropyLoss() 
learning_rate = 0.01
optimizer= optim.SGD(model.parameters(), lr=learning_rate)    

# Train model
iter = 0 
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        #Load images in Variables
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        iter = iter + 1
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))













model.eval()
correct=0
total=0
for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))       