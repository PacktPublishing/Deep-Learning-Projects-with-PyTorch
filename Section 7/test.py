

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch import optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
data_transforms = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir='images'
image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(
        dataset=image_datasets,
        batch_size=2, shuffle=True)

image_datasets.classes
labels_h = ('Circle','Rectangle','Triangle')
dataset_size = len(image_datasets) # In our case 7 images

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        #Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, 
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
                
        self.fc1 = nn.Linear(32*128*128, 7)
    
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
learning_rate = 0.001
optimizer= optim.SGD(model.parameters(), lr=learning_rate)    

# Train model

num_epochs = 5
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
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(image_datasets)//dataset_size, loss.data[0]))


data_dir='train'
test_image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(
        dataset=test_image_datasets,
        batch_size=1, shuffle=True)



model.eval()
correct=0
total=0

for i, (images,labels) in enumerate(test_loader):        
        imshow(torchvision.utils.make_grid(images))
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
       
        print("Prediction --  ",labels_h[predicted[0]])
        


