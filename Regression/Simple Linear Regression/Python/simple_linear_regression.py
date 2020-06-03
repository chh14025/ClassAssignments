#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:40:24 2020

@author: s.p.
"""


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

os.getcwd()
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#visualising Train
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Experiences (Training)')
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()

#Test
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Experiences (Test)')
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()