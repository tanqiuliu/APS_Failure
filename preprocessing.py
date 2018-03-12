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

def imputation(train_X, test_X, method='median'):
    if(method == 'mean'):
        train_X = train_X.fillna(train_X.mean())
        test_X = test_X.fillna(train_X.mean())
    elif(method == 'median'):
        train_X = train_X.fillna(train_X.median())
        test_X = test_X.fillna(train_X.median())
    else:
        print('Invalid imputation method!') 
    return train_X, test_X


def get_nan_col(train_X, nan_threshold=0.1):
    # remove columns with missing values more than threshold 
    nan_col = []
    num_null = dict(train_X.isnull().sum())
    for feat in num_null:
        if(num_null[feat] > nan_threshold * train_X.shape[0]):
            nan_col.append(feat)
    return nan_col


def get_zero_col(train_X, zero_threshold=0.9):
    # remove columns with zero-value 
    zero_col = []
    num_zero = dict((train_X == 0).sum(axis = 0))
    for feat in num_zero:
        if(num_zero[feat] > zero_threshold * train_X.shape[0]):
            zero_col.append(feat)
    return zero_col

def remove_columns(train_X, test_X, col_remove):
    train_X = train_X.drop(columns=col_remove)
    test_X = test_X.drop(columns=col_remove)
    print("Features dropped: %s." %col_remove)
    return train_X, test_X


def remove_records(train_X, train_y, threshold=0.1):
    # remove records with too much missing values 
    train_X = pd.concat([train_X, train_y], axis=1)
    num_null = dict(train_X.isnull().sum(axis=1))
    row_remove = []
    for r in num_null:
        if(num_null[r] > threshold * train_X.shape[1]):
            row_remove.append(r)
    train_X = train_X.drop(train_X.index[row_remove])
    print("%s records are removed." %len(row_remove))
    train_y = pd.DataFrame.copy(train_X['class'])
    train_X = train_X.drop(columns=['class'])
    return train_X, train_y


def remove_outliers(train_X):
    pass