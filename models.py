#!/usr/bin/python
# -*- coding: utf-8 -*-

import sklearn
from utils import *
from preprocessing import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

def evaluate(X, y, mdl):
    tcost = make_scorer(total_cost,greater_is_better=True)
    n_fold = 5
    return np.mean(cross_val_score(mdl, X, y, scoring=tcost, cv=n_fold,n_jobs=4))


