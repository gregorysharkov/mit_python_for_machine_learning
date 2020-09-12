# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:28:52 2020

@author: Grigory
"""
import numpy as np

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code her
    return x*y if x<=y else x/y
    raise NotImplementedError
    
print(scalar_function(2,3) == 6)
print(scalar_function(3,2) == 1.5)

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    vfunc = np.vectorize(scalar_function)
    return vfunc(x,y)
    raise NotImplementedError
    
