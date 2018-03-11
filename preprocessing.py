#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import *

# Try difference methods to do deal with missing values
# 1. Removing columns
# 2. Removing records
# 3. Imputation using mean/median...
# 4. Collaborative filtering to fill the nan entries
# 5. Multiple Imputation

# Try to detect and remove anomlies before applying imputations

def imputation(train_X, test_X, method='mean'):
    if(method == 'mean'):
        all_X = pd.concat([train_X, test_X], axis=0)
        all_X = all_X.fillna(all_X.mean())
    elif(method == 'median'):
        all_X = pd.concat([train_X, test_X], axis=0)
        all_X = all_X.fillna(all_X.median())
    else:
        print('Invalid imputation method!') 
    train_X = all_X.iloc[0:train_X.shape[0]]
    test_X = all_X.iloc[train_X.shape[0]:]
    return train_X, test_X


def removing_columns(train_X, test_X, threshold=0.1):
    # removing columns with missing values more than threshold 
    all_X = pd.concat([train_X, test_X], axis=0)
    num_null = dict(all_X.isnull().sum())
    col_remove = []
    for feat in num_null:
        if(num_null[feat] > threshold * all_X.shape[0]):
            col_remove.append(feat)
    all_X = all_X.drop(columns=col_remove)
    train_X = all_X.iloc[0:train_X.shape[0]]
    test_X = all_X.iloc[train_X.shape[0]:]
    return train_X, test_X
