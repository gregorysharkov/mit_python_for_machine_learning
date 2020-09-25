# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:27:00 2020

@author: Grigory
"""
import numpy as np
from string import punctuation, digits
import random

""" feature_matrix = np.array([
    [ 0.1837462,   0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835,   0.21370063,  0.33344998, -0.40850817, -0.13105809],
    [ 0.08254096,  0.06012654,  0.19821234,  0.40958367,  0.07155838, -0.49830717,   0.09098162,  0.19062183, -0.27312663,  0.39060785],
    [-0.20112519, -0.00593087,  0.05738862,  0.16811148, -0.10466314, -0.21348009,   0.45806193, -0.27659307,  0.2901038,  -0.29736505],
    [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345,  0.07636098,  -0.42087389, -0.16322197, -0.02759763,  0.0297091 ],
    [-0.18082261,  0.28644149, -0.47549449, -0.3049562,   0.13967768,  0.34904474,   0.20627692,  0.28407868,  0.21849356, -0.01642202]
])
labels = [-1,-1,-1, 1,-1]
L = 0.1456692551041303
T = 10 """


""" feature_matrix = np.array([
    [ 0.32453673,  0.06082212,  0.27845097,  0.27124962, -0.48858134],
    [-0.07490036, -0.2226942,   0.46808161, -0.15484728, -0.06555043],
    [ 0.48089473,  0.11053774, -0.39253255, -0.45844357,  0.19818921],
    [ 0.39728286,  0.14426349,  0.23446484, -0.46963688,  0.30978055],
    [-0.2836313,   0.20048277,  0.10600686, -0.47812081,  0.24772569],
    [-0.38813183, -0.39082381,  0.02482903,  0.46576666, -0.22720277],
    [ 0.15482689, -0.16083218,  0.38637948, -0.14209394,  0.05076824],
    [-0.1238048,  -0.1064888,  -0.28800396, -0.47983335,  0.31652173],
    [ 0.31485345,  0.30679047, -0.1907081,  -0.0961867,   0.27954887],
    [ 0.4024408,   0.2990748,   0.34148516, -0.311256,    0.13324454]
])
labels = [-1, -1,  1,  1,  1,  1, -1, -1,  1,  1]
T=10
L = 0.705513226934028
"""

feature_matrix = np.array([[-0.46060257,  0.0047093,  -0.47889493, -0.48770436,  0.34646198, -0.40440974,   0.08502949, -0.08478305,  0.21789522, -0.0033492 ]])
labels = [-1]
T=500
L = 0.43351443489540376


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(np.dot(current_theta,feature_vector) + current_theta_0) < 1.000001:
        new_theta = (1-eta*L)*current_theta + eta*label*feature_vector
        new_theta_0 = current_theta_0 + eta*label
    else:
        new_theta = (1-eta*L)*current_theta
        new_theta_0 = current_theta_0
    
    return new_theta, new_theta_0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    counter = 0

    for _ in range(T):
        for i in get_order(feature_matrix.shape[0]):
            counter += 1
            eta = 1/(counter**(1/2))
            theta, theta_0 = pegasos_single_step_update(feature_vector=feature_matrix[i],
                                                        label = labels[i],
                                                        L = L,
                                                        eta = eta,
                                                        current_theta = theta, 
                                                        current_theta_0 = theta_0)
        pass
    pass
    return theta, theta_0

print(pegasos(feature_matrix, labels, T, L))
