# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:30:27 2020

@author: Grigory
"""

import numpy as np

x = [[-1,-1],[1,0],[-1,1.5]]
y = [1,0,1]

alt_x = [[1,0],[-1,1.5],[-1,-1]]
alt_y = [0,1,1]


def perceptron(features,labels, num_iter):
    # set weights to zero
    w = np.zeros(shape=(len(features)-1))
    weights = []
    misclassified_ = [] 
  
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            #print(x)
            x = np.array(x)
            #x = np.insert(x,0,1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else 0.0
            
            delta = (label - target)

        
            if(delta): # misclassified
                misclassified += 1
                w += (delta * x)
                weights.append(list(w.copy()))
            #print(f"y: {y}, w: {w}, x: {x}, delta: {delta}, miss: {misclassified}")  
             
        misclassified_.append(misclassified)
    return (w, misclassified_, weights)
             
num_iter = 5
w, misclassified_, weights = perceptron(x, y, num_iter)
print(misclassified_, weights)
w, misclassified_, weights = perceptron(alt_x, alt_y, num_iter)
print(misclassified_,weights)
#print (w,misclassified_)