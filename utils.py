#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_data():
    # default data path
    train_path = './data/aps_failure_training_set.csv'
    test_path = './data/aps_failure_test_set.csv'
    # load
    train_X = pd.read_csv(train_path, skiprows=20)
    test_X = pd.read_csv(test_path, skiprows=20)
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


def total_cost(y_true, y_pred):
    # y_true, y_pred: numpy array with 1 and 0
    FP = (y_true != y_pred) * y_pred
    FN = (y_true != y_pred) * y_true
    return 10 * np.sum(FP)/len(y_true) + 500 * np.sum(FN)/len(y_true)


def metrics(y_true, y_pred):
    print("accuracy score: %s" %accuracy_score(y_true, y_pred))
    print("precision: %s" %precision_score(y_true, y_pred))
    print("recall: %s" %recall_score(y_true, y_pred))
    print("total_cost: %s" %total_cost(y_true, y_pred))

