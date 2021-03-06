#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from preprocessing import *

def load_data(shuffle = False):
    # default data path
    train_path = './data/aps_failure_training_set.csv'
    test_path = './data/aps_failure_test_set.csv'
    # load
    train_X = pd.read_csv(train_path, skiprows=20)
    test_X = pd.read_csv(test_path, skiprows=20)
    if(shuffle == True):
        train_X = shuffle(train_X)
        test_X = shuffle(test_X)
    # X,y split
    train_y = pd.DataFrame.copy(train_X['class'])
    test_y = pd.DataFrame.copy(test_X['class'])
    train_X = train_X.drop(columns=['class'])
    test_X = test_X.drop(columns=['class'])
    # basic transform of dtype
    train_X = train_X.replace('na', np.NaN).astype(np.float64)
    test_X = test_X.replace('na', np.NaN).astype(np.float64)
    train_y = train_y.replace(['neg','pos'],[0,1])
    test_y = test_y.replace(['neg','pos'],[0,1])
    return train_X, train_y, test_X, test_y


def total_cost(y_true, y_pred, threshold=0.01667):
    FP = ((y_pred > threshold) & (y_true == 0)).sum()
    FN = ((y_pred <= threshold) & (y_true == 1)).sum()
    return 10 * FP + 500 * FN


def metrics(y_true, y_pred):
    print("accuracy score: %s" %accuracy_score(y_true, y_pred))
    print("precision: %s" %precision_score(y_true, y_pred))
    print("recall: %s" %recall_score(y_true, y_pred))
    print("total_cost: %s" %total_cost(y_true, y_pred))


def evaluate(X, y, mdl):
    tcost = make_scorer(total_cost,greater_is_better=True)
    n_fold = 5
    return np.mean(cross_val_score(mdl, X, y, scoring=tcost, cv=n_fold,n_jobs=4))


def get_FP(X, y_true, y_pred):
    return np.where((y_pred >  0) & (y_true == 0))

def get_FN(X, y_true, y_pred):
    return np.where((y_pred <=  0) & (y_te == 1))