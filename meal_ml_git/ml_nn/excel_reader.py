#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:23:28 2022

@author: rodrigocampos
"""

# This script collects the data from the excel file and
# outputs it in a nice format for the machine learning script

import pandas as pd

def read_excel(file_path : str):
    inputs = []
    outputs = []
    
    df = pd.read_excel(file_path,index_col=0)
    df = df.reset_index()  # make sure indexes pair with number of rows
    i = 0
    for index, row in df.iterrows():
        inputs.append([])
        outputs.append([])
        inputs[i].append([])
        outputs[i].append([])
        # turn categorical data in numerical data
        if row['Type'] == 'breakfast':
            inputs[i][0].append(1)
        elif row['Type'] == 'lunch':
            inputs[i][0].append(2)
        elif row['Type'] == 'dinner':
            inputs[i][0].append(3)
        else:
            inputs.pop()
            outputs.pop()
            continue
        
        if row['gender'] == 'male':
            inputs[i][0].append(1)
        elif row['gender'] == 'female':
            inputs[i][0].append(2)
        else:
            inputs[i][0].append(3)

        for title in df.columns:
            if title not in ['index','Type','gender','Score(1:worst 2:bad 3:good 4:best)']:
                inputs[i][0].append(row[title])
        outputs[i][0].append(row['Score(1:worst 2:bad 3:good 4:best)'])
        
        i+=1
    #print(inputs)
    return df.columns, inputs, outputs

def minmax(inputs):
    min_values = [0 for _ in range(len(inputs[0][0]))]
    max_values = [0 for _ in range(len(inputs[0][0]))]
    for item in inputs:
        for i,value in enumerate(item[0]): 
            if value > max_values[i]:
                max_values[i] = value
            elif value < min_values[i] or min_values[i] == 0:
                min_values[i] = value
                
    for i,item in enumerate(inputs):
        for j,value in enumerate(item[0]):
            inputs[i][0][j] -= min_values[j]
            inputs[i][0][j] /= max_values[j] - min_values[j]
    return inputs

if __name__ == '__main__':
    _, inputs, _ = read_excel(r'input_file/input_file.xlsx')
    minmax(inputs)
    #print(inputs)
    