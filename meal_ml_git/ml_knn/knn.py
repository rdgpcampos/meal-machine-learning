#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:55:32 2022

@author: rodrigocampos
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import excel_reader
import numpy as np
from importlib import reload

excel_reader = reload(excel_reader)    

def knn():
    # getting data
    x_total, y_total, labels = excel_reader.get_data(r'../input_file/input_file.xlsx')
    
    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.10)
    
    # train and testing
    clf = KNeighborsClassifier(p=1)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(accuracy_score(y_test, predictions))
    
    # numerical output range
    xx, yy = np.meshgrid(np.arange(0, 1, 0.2),
                         np.arange(0, 1, 0.1))
    # categorical output range
    cat_xx, cat_yy = np.meshgrid(np.arange(0, 4, 1),
                                 np.arange(0, 4, 1))
    
    plot_flag = input('Plot figures? [y/n]')
    if plot_flag != 'y':
        return
    
    # plot data
    plt.figure()   
    for i,title_x in enumerate(x_train.columns):
        for j,title_y in enumerate(x_train.columns):
            if j >= i:
                continue
            plt.scatter(x_train[title_x], 
                        x_train[title_y], 
                        c=y_train, 
                        cmap = ListedColormap(['#FF770A',
                                               '#D65225',
                                               '#88B04B',
                                               '#178512']))
            
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
    
            if i in [0,1]:
                plt.xlim(0, 4)
            else:
                plt.xlim(0,1)
            if j in [0,1]:
                plt.ylim(0, 4)
            else:
                plt.ylim(0, 1)
    
            plt.title("4-Class classification (k = 5)")
            plt.show()
    return

if __name__ == '__main__':
    knn()


