#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:03:58 2020

@author: s.p.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y= dataset.iloc[:, -1:].values

''' No training split is required because the data set is small
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .1, random_state = 0)
'''

'''No Feature Scailing is required because polynominal regression consists of adding polynominal
    terms into the linear regression equation
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform()
x_test = sc_x.transform(x_test)
'''

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualizing the regression models
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')

#Predicting a new result with linear regression
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


