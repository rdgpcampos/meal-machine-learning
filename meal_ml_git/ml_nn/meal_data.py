#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import my_nn as mnn
import excel_reader
from importlib import reload
from math import floor
import numpy as np

excel_reader = reload(excel_reader)   

def get_net():
    # training data
    column_titles, x_total, y_total = excel_reader.read_excel(r'../input_file/input_file.xlsx')
    x_total = excel_reader.minmax(x_total)
    
    # splitting train data 
    train_size = floor(len(x_total)*0.9)
    
    x_train = np.array(x_total[:train_size])
    x_test = np.array(x_total[train_size:])
    
    y_train = np.array(y_total[:train_size])
    y_test = np.array(y_total[train_size:])
    
    # setting up different networks with different initial values
    net1, net2, net3 = mnn.Network(), mnn.Network(), mnn.Network()
    net_list = [net1,net2,net3]
    rate_list = [0 for _ in range(len(net_list))]
    index_of_best_net = 0
    max_rate = 0
    
    for i,net in enumerate(net_list,start=1):
        
        # network architecture
        net.add(mnn.FCLayer(16, 64))
        net.add(mnn.ActivationLayer(mnn.leaky_relu, mnn.leaky_relu_derivative))
        net.add(mnn.FCLayer(64, 128))
        net.add(mnn.ActivationLayer(mnn.leaky_relu, mnn.leaky_relu_derivative))
        net.add(mnn.FCLayer(128, 16))
        net.add(mnn.ActivationLayer(mnn.leaky_relu, mnn.leaky_relu_derivative))
        
        # train
        net.use(mnn.mse, mnn.mse_derivative)
        net.fit(x_train, y_train, epochs=1000, learning_rate=0.01)
        
        # test accuracy
        out = net.predict(x_test)
        rate = 0
        size = 0
        
        # pick rounded value of largest output
        for j,test in enumerate(out):
            if round(max(test[0])) == y_test[j][0][0]:
                rate += 1
            size += 1
        rate /= size
        rate_list.append(rate)
        if rate > max_rate:
            max_rate = rate
            index_of_best_net = i-1
        print("\n")
        print("prediction rate:")
        print(rate)
        
    # get best network out of list
    best_net = net_list[index_of_best_net]
    
    return best_net

if __name__ == "__main__":
    get_net()