# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:27:00 2020

@author: Grigory
"""
import numpy as np
from string import punctuation, digits
import random
import project1 as p1
import utils

# toy_features, toy_labels = toy_data = utils.load_toy_data("sentiment_analysis/toy_data.tsv")
# T = 10
# L = 0.2

# thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
# thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)
# print(thetas_pegasos[0])
# print(thetas_pegasos[1])
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    eps = 1e-8
    
    # if abs(agreement) < eps or agreement < 0:   # 1st condition to check if = 0
    new_theta = current_theta.copy()
    new_theta_0 = current_theta_0
    agreement = float(label*(new_theta.dot(feature_vector)+new_theta_0))

    if abs(agreement) < eps or agreement < 0:   # 1st condition to check if = 0
            new_theta = current_theta.copy() + label*feature_vector
            new_theta_0 = current_theta_0 + label
            print(f"\tlabel: {label}")
            print(f"\tagreement: {current_theta.dot(feature_vector)+ 0}")
            print(f"\tUpdating theta. New theta: {new_theta}")
            
    return (new_theta, new_theta_0)
    
#pragma: coderesponse end
#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    current_theta_0 = 0.0
    current_theta = np.zeros(feature_matrix.shape[1])
    
    for t in range(T):
        print(f"Starting round {t+1}")
        for i in range(feature_matrix.shape[0]): #get_order(feature_matrix.shape[0]):
            print(f"element {i+1}")
            current_theta, current_theta_0 = \
            perceptron_single_step_update(feature_matrix[i,:], labels[i], \
                                          current_theta, current_theta_0)
            
    return (current_theta, current_theta_0)
#pragma: coderesponse end

feature_matrix = np.array([[-1,1],[1,-1],[1,1],[2,2]])
labels = np.array([1,1,-1,-1])

theta = perceptron(feature_matrix, labels, 3)
print(theta)