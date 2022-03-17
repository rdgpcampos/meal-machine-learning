#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:07:36 2022

@author: rodrigocampos
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import excel_reader
import matplotlib.pyplot as plt
from importlib import reload

excel_reader = reload(excel_reader)

# get data
feature_names, x, y = excel_reader.read_excel(r'../input_file/input_file.xlsx')

# splitting test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

# train
classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(x_train, y_train)

# test
y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot
plt.figure(figsize=(60,30))
plot_tree(classifier,filled=True,fontsize=10,feature_names=feature_names)
plt.savefig('tree.png',format='png',bbox_inches = "tight")