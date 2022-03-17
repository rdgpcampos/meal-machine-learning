#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:09:06 2022

@author: rodrigocampos
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

@dataclass
class FCLayer:
    input_size: int = 0
    output_size: int = 0
    weights: float = field(init=False)
    bias: float = field(init=False)
    input_data: float = field(init=False)
    output_data: float = field(init=False)

    def __post_init__(self):
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5

    # returns output for a given input
    def forward(self,input_data):
        self.input_data = input_data
        # Y = XW + B
        self.output_data = np.dot(self.input_data,self.weights) + self.bias
        return self.output_data
    # computes dE/dW, dE/dB for a given output error dE/dY. 
    # returns input error dE/dX
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input_data.T,output_error)
        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
@dataclass
class ActivationLayer:
    activation: Callable[[float],float]
    activation_derivative: Callable[[float],float]
    input_data: float = field(init=False)
    output_data: float = field(init=False)
    
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = self.activation(self.input_data)
        return self.output_data
    
    def backward(self, output_error, learning_rate):
        return self.activation_derivative(self.input_data)*output_error