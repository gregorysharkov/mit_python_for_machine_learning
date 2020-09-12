# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:18:41 2020

@author: Grigory
"""
import numpy as np
def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    return(np.tanh(np.matmul(np.transpose(weights),inputs)))
    raise NotImplementedError
    
inputs = np.array([1,2])
weights = np.array([1,.5])
print(neural_network(inputs, weights))