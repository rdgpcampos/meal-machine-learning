#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:28:32 2022

@author: rodrigocampos
"""
import numpy as np

def leaky_relu(x: float, alpha = 0.01)->float:
    return np.where(x > 0, x, x*alpha)

def  leaky_relu_derivative(x: float, alpha = 0.01)->float:
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx