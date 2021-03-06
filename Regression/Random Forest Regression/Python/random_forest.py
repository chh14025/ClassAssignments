#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:59:23 2020

@author: s.p.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 3000, random_state = 0)
regressor.fit(x, y)


y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x),max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color= 'red')
plt.plot(x_grid,regressor.predict(x_grid))

