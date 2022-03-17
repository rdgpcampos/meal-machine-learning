#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;