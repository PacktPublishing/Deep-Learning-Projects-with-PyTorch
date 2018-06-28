# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:56:00 2018

@author: Ashish Bhatia
"""

# AutoEncoders

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
number_of_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
number_of_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for user in range(1, number_of_users + 1):
        movieid = data[:,1][data[:,0] == user]
        ratingid = data[:,2][data[:,0] == user]
        ratings = np.zeros(number_of_movies)
        ratings[movieid - 1] = ratingid
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set) 

# Creating the architecture of the Neural Network


class AutoEncoder(nn.Module):
    def __init__(self, ):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(number_of_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, number_of_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
autoencoder = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(autoencoder.parameters(), lr = 0.01, weight_decay = 0.5)



nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for user in range(number_of_users):
        input = Variable(training_set[user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = autoencoder(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = number_of_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
















# Testing the autoencoder
test_loss = 0
s = 0.
for user in range(number_of_users):
    input = Variable(training_set[user]).unsqueeze(0)
    target = Variable(test_set[user])
    if torch.sum(target.data > 0) > 0:
        output = autoencoder(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = number_of_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))








