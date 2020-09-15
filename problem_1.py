# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:31:46 2020

@author: Grigory
"""

import numpy as np

x = np.array([[-1,-1],[1,0],[-1,1.5]])
y = np.array([1,-1,1])

alt_x = np.array([[1,0],[-1,1.5],[-1,-1]])
alt_y = np.array([1,-1,1])

def perception(x,y):
    theta = np.array([0.,0.])
    all_thetas = []
    theta_changed = True
    i=0
    while theta_changed:
        theta_changed=False
        print(f"Iteration {i+1}")
        for t in range(len(x)):
            check = y[t]*theta*x[t]
            print(y[t],theta,x[t],check.sum())
            if check.sum() <= 0:
                theta += y[t]*x[t]
                all_thetas.append(list(theta.copy()))
                theta_changed = True
        i+=1
    return theta, all_thetas

def predict(x,y):
    theta, all_thetas = perception(x,y)
    print("Predicting...")
    print(f"\ttheta: {theta}\n\tall_thetas: {all_thetas}")
    for i in range(len(x)):
        prediction = x[i]*theta
        print(f"\tx_{i+1}: {prediction.sum()>=0}")

predict(x,y)
predict(alt_x,alt_y)
