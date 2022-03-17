#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:02:38 2022

@author: rodrigocampos
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def read_excel(file_path : str):
    inputs = []
    outputs = []
    encoder = OrdinalEncoder()
    
    df = pd.read_excel(file_path,index_col=0)
    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df = df.reset_index()  # make sure indexes pair with number of rows
    df[["Type","gender"]] = encoder.fit_transform(df[["Type","gender"]])
    
    inputs = df.drop(['index','Score(1:worst 2:bad 3:good 4:best)'],axis=1)
    
    outputs = df['Score(1:worst 2:bad 3:good 4:best)']

    return list(df.columns[1:17]), inputs, outputs

if __name__ == "__main__":
    read_excel(r'input_file/input_file.xlsx')