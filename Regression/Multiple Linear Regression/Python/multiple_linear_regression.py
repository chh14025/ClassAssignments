#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:23:24 2020

@author: s.p.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(),[3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
print (x)
#avoid the dummy variable trap

x=x[:,1:]
#Python takes care of it, so this step is unncecessary in python


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)

# Building the optimal model using backward elimination
import statsmodels.api as sm
#x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1).astype(float)
x = sm.add_constant(x).astype(float)
x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3,4,5]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3,5]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

''' 
#- This is an automatic implementation of backward elimination with p-values
def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range (0, numVars):
        regressor_ols = sm.OLS(y,x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars-i):
                if (regressor_ols.pvalues[j].astype(float) == x = np.delete(x,j,1)):
    regressor_ols.summary()
    return x
SL = 0.5
x_opt = x_opt = x[:, [0,1,2,3,4,5]]
x_model = backwardElimination (x_opt, SL)
'''


