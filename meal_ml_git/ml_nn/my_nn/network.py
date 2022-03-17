#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output_data = input_data[i]
            for layer in self.layers:
                output_data = layer.forward(output_data)
            result.append(output_data)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate,print_flag=False):
        # sample dimension first
        samples = len(x_train)
        #print("samples :", samples)
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output_data = x_train[j]
                for layer in self.layers:
                    output_data = layer.forward(output_data)
                #print("err :",err)
                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output_data)

                # backward propagation
                error = self.loss_derivative(y_train[j], output_data)
                #print("error :",error)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            if print_flag:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
