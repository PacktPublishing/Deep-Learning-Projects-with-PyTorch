#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:25:15 2018

@author: ashish
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel

import numpy as np
import pandas as pd


# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', 
                     header=None, engine='python', 
                     encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', sep='::', 
                     header=None, engine='python', 
                     encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', 
                     header=None, engine='python')

# Prepare training set and test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') 
training_set = np.array(training_set, dtype='int')


test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t') 
test_set = np.array(test_set, dtype='int')

# Getting the numbers of users and movies
total_no_of_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
total_no_of_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data in to matrix with user and movies columns

def convert(data):
    new_data = []
    for user_id in range(1, total_no_of_users+1):
        movie_ids = data[:,1][data[:,0]==user_id]
        rating_ids = data[:,2][data[:,0]==user_id]
        ratings = np.zeros(total_no_of_movies)
        ratings[movie_ids -1 ] = rating_ids
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Convert it in Torch Tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 
# 1 (Liked) or 0 (Not Liked)
# 
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, input_nodes, hidden_nodes):
        self.weight = torch.randn(hidden_nodes, input_nodes)
        self.hidden_bias = torch.randn(1, hidden_nodes)
        self.input_bias = torch.randn(1, input_nodes)
    def sample_hidden(self, x):
        # Input -> Input X weight + bias
        wx = torch.mm(x, self.weight.t())
        activation = wx + self.hidden_bias.expand_as(wx)
        prob_hidden_given_visible = torch.sigmoid(activation)
        return prob_hidden_given_visible, torch.bernoulli(prob_hidden_given_visible)
    def sample_input(self, y):
        wy = torch.mm(y, self.weight)
        activation = wy + self.input_bias.expand_as(wy)
        prob_visbile_given_hidden = torch.sigmoid(activation)
        return prob_visbile_given_hidden, torch.bernoulli(prob_visbile_given_hidden)
    # V0 = Input vector containing the ratings of all the movies by one user , we loop over all the user
    # Vk = Visible node obtainied after k iteration / sampling  k round trips from visible node to hidden node  first and then way back from hidden nodes to viisble nodes thats 
      # visible node obltained after the k iteration in case contrastive divergence 
    # ph0 = Vector of probaliities that the first iteration the hidden nodes equal one given the values of zero. Input vector of observation
    # phk = Probabilities of hidden node after k sampling give the values of visible nodes we get.

    def train(self, visible0, visiblek, probhidden0, prodhiddenk):
        self.weight += torch.mm(visible0.t(), probhidden0) - torch.mm(visiblek.t(), prodhiddenk)
        self.input_bias += torch.sum((visible0 - visiblek), 0)
        self.hidden_bias += torch.sum((probhidden0 - prodhiddenk), 0)
        

input_nodes = len(training_set[0]) # Number of visible nodes
hidden_nodes = 100  # Number of hidden nodes
batch_size = 100
rbm = RBM(input_nodes, hidden_nodes)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # For float it will used to normalize the loss
    for user_id in range(0, total_no_of_users - batch_size, batch_size):
        vk = training_set[user_id:user_id+batch_size]
        v0 = training_set[user_id:user_id+batch_size]
        ph0,_ = rbm.sample_hidden(v0) # Using Visible Node
        for k in range(10): # K steps of contrastive divergence
            _,hk = rbm.sample_hidden(vk)
            _,vk = rbm.sample_input(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))














# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(total_no_of_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_input(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))














