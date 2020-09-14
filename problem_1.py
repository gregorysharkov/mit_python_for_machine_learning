# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:31:46 2020

@author: Grigory
"""

import numpy as np

x = np.array([[-1,-1,1],[1,0,1],[-1,1.5,1]])
y = np.array([1,-1,1])

def perception(x,y):
    theta = np.array([0.,0.,0.])
    for t in range(len(x)):
        print(f"Step {t+1}:")
        print(f"\tOriginal theta: {theta}")
        check = y[t]*theta*x[t].sum()
#        print(f"\ty[t]*theta*x[t]: {check}")
        print(f"\tcheck: {check.sum()}")
        if check.sum() <= 0:
            theta += y[t]*x[t]
            print(f"\tUpdating theta, new theta: {theta}")
    
    print(f"Final theta: {theta}")
    return theta

def predict(x,y):
    theta = perception(x,y)
    print("Predicting...")
    print(f"\ttheta: {theta}")
    for i in range(len(x)):
        prediction = x[i]*theta
        print(f"\tx_{i+1}: {prediction.sum()}")

alt_x = np.array([[1,0,1],[-1,1.5,1],[-1,-1,1]])
alt_y = np.array([1,-1,1])

predict(x,y)
predict(alt_x,alt_y)
