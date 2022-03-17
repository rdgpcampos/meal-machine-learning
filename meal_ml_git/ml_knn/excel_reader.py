#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 00:17:44 2022

@author: rodrigocampos
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer

def read_excel(file_path : str):
    df = pd.read_excel(file_path,index_col=0)
    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df = df.reset_index()  # make sure indexes pair with number of rows
    
    inputs = df.drop(['index','Score(1:worst 2:bad 3:good 4:best)'],axis=1)
    
    categorical = ['Type','gender']
    numerical = ['age',
                 'height',
                 'weight',
                 'EER[kcal]',
                 'P target(15%)[g]',
                 'F target(25%)[g]',
                 'C target(60%)[g]',
                 'number of dishes',
                 'E[kcal]',
                 'P[g]',
                 'F[g]',
                 'C[g]',
                 'Salt[g]',
                 'Vegetables[g]']
    
    outputs = df['Score(1:worst 2:bad 3:good 4:best)']

    return inputs, outputs, numerical, categorical

def preprocess(inputs, outputs, numerical, categorical):
    labels = []
    for label in categorical:
        labels.append(label)
    for label in numerical:
        labels.append(label)
    
    column_transformer = make_column_transformer(
    (OrdinalEncoder(), categorical),
    (MinMaxScaler(), numerical),
    remainder='passthrough')

    inputs = column_transformer.fit_transform(inputs)
    inputs = pd.DataFrame(data=inputs)
    
    return inputs, outputs, labels

def get_data(file_path : str):
    inputs, outputs, numerical, categorical = read_excel(file_path)
    return preprocess(inputs, outputs, numerical, categorical)

if __name__ == "__main__":
    get_data(r'input_file/input_file.xlsx')