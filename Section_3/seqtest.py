#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:36:03 2018

@author: ashish
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:36:45 2018

@author: Ashish Bhatia
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


idx2char = ['a','b','c','d','e', 'f']

# Teach acdbef -> bdface

x_data = [[0,2,3,1,4,5]] # acdbef
x_one_hot = [[1,0,0,0,0,0], 
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [0,1,0,0,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]
        ]

y_data = [1,3,5,0,2,4] # bdface

inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 6
input_size = 6 
hidden_size = 6
batch_size = 1 # One Sentence
sequence_length = 1 # Let go one by one 
num_layers = 1



class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True)
    
    def forward(self, x, hidden, cell):
        x = x.view(batch_size, sequence_length, input_size)
        out, (hidden, cell) = self.lstm(x, hidden, cell)
        out= out.view(-1, num_classes)
        return hidden,out, cell

    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))

model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    cell = model.init_hidden()
    print("Predicted String")
    for ins, label in zip(inputs,labels):
        hidden, output, cell = model(ins, hidden, cell)
        val, idx = output.max(1)
        print(idx2char[idx.data[0]])
        loss+=criterion(output, label)
    
    print(", epoch: %d, loss: %1.3f" % (epoch+1, loss.data[0]))

    loss.backward()
    optimizer.step()













cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

inputs = Variable(torch.Tensor([['h','e','l','l','o']]))

print("Input Size", inputs.size())

hidden = Variable(torch.randn(1,1,2))

out, hidden = cell(inputs, hidden)

print (out.data)