#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:13:43 2020

@author: s.p.
"""

# SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y= dataset.iloc[:, -1:].values

''' No training split is required because the data set is small
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .1, random_state = 0)
'''

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)


y_pred = regressor.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

#Visualizing the regression models
plt.scatter(x,y, color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.show()