#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:53:52 2018

@author: ashish
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

data = pd.read_csv('creditcard.csv')
print (data)
fraud_case = data[data['Class']==1]
ok_case = data[data['Class']==0]

plt.plot(data['Time'], data['Amount'])


groups = data.groupby('Class')
fig, ax = plt.subplots()
for name,group in groups:
    ax.plot(group.Time, group.Amount, marker='o', linestyle='', ms=5, label=name)
ax.legend()
plt.show()

data['Time'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24.)
scl = StandardScaler()
data['Time'] = scl.fit_transform(data['Time'].values.reshape(-1,1))
scl = StandardScaler()
data['Amount'] = scl.fit_transform(data['Amount'].values.reshape(-1,1))

x_train, x_test = train_test_split(data, test_size = 0.2, random_state = 42)
x_train.shape
x_test.shape

x_train = x_train[x_train['Class'] == 0]
x_train.shape
x_train = x_train.drop('Class', axis = 1)
x_train.shape
y_test = x_test['Class'].values
y_test.shape
x_test = x_test.drop('Class', axis = 1)
x_test = x_test.values
x_train = x_train.values

x_train.shape, x_test.shape



train_tensor = torch.FloatTensor(x_train)
test_tensor = torch.FloatTensor(x_test)
train_loader = DataLoader(train_tensor,batch_size = 1000)
test_loader = DataLoader(test_tensor, batch_size = 1000)



class AutoEncoder_Fraud(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,30)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return (x)
    
    
model = AutoEncoder_Fraud()
loss=nn.MSELoss()
learning_rate = 1e-2
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(30):
    losses=[]
    train_data_loader = iter(train_loader)
    for t in range(len(train_data_loader)):
        data = next(train_data_loader)
        data_v = Variable(data)
        y_pred = model(data_v)
        l = loss(y_pred,data_v)        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('Epoch : {%s} Loss : {%s}' % (epoch, l.data[0]))
    
    
    
    
test_data_loader = iter(test_loader)
preds = []
for t in range(len(test_data_loader)):
    data = next(test_data_loader)
    data_v = Variable(data)
    y_pred = model(data_v)
    print("Loss -> ", loss(y_pred,data_v.data[0]))
