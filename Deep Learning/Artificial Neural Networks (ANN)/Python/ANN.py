#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:21:28 2020

@author: s.p.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1]
y = dataset.iloc[:,-1:].values

categorical = ["Gender", "Geography"]
for names in categorical:
    x = pd.get_dummies(x,columns = [names],drop_first=True)
x = x[:].values
    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#activation = soft_max for more than two dependent variable

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#categorical_crossentropy for more than two categories

classifier.fit(x_train,y_train, batch_size= 10, nb_epoch=100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm
sns.heatmap(cm,annot=True)
ac = accuracy_score(y_test, y_pred)
ac










