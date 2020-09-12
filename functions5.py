# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 22:26:47 2020

@author: Grigory
"""

def get_sum_metrics(predictions, metrics=[]):
    if len(metrics) > 0:
        metrics = [predictions]
    for i in range(3):
        element = predictions + i
        metrics.append(element)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric

    return sum_metrics

print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15