#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import *
from preprocessing import *
from sklearn.tree import DecisionTreeRegressor

# load data
(train_X, train_y, test_X, test_y) = load_data()

# remove columns
nan_col = get_nan_col(train_X)
zero_col = []
# zero_col = get_zero_col(train_X)
col_remove = list(set(nan_col+zero_col))
#col_remove = []
X_tr, X_te = remove_columns(train_X, test_X, col_remove)

# imputation
X_tr, X_te = imputation(X_tr, X_te,method='median')
X_tr, X_te = X_tr.values, X_te.values
# normalize distribution
# quantile_transformer = sklearn.preprocessing.QuantileTransformer(random_state=0)
# X_tr = quantile_transformer.fit_transform(X_tr)
# X_te = quantile_transformer.transform(X_te)

y_tr, y_te = train_y.values, test_y.values

# model
cart = DecisionTreeRegressor(min_samples_leaf=70,max_depth=30)
cart = cart.fit(X_tr, y_tr)
y_pred = cart.predict(X_te) 

print("Total_cost: %s" %total_cost(y_te, y_pred>0))
print("Precision: %s" %precision_score(y_te, y_pred>0))
print("Recall: %s" %recall_score(y_te, y_pred>0))