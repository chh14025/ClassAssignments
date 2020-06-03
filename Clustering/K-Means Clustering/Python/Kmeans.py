#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:07:43 2020

@author: s.p.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#The Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.ylabel ('WCSS')
tick = np.arange(1,11,1)
plt.xticks(ticks = tick)
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans ==0,0], x[y_kmeans ==0, 1], s = 100, c = 'red', label='Careful')
plt.scatter(x[y_kmeans ==1,0], x[y_kmeans ==1, 1], s = 100, c = 'blue', label='Standard')
plt.scatter(x[y_kmeans ==2,0], x[y_kmeans ==2, 1], s = 100, c = 'green', label='Target')
plt.scatter(x[y_kmeans ==3,0], x[y_kmeans ==3, 1], s = 100, c = 'cyan', label='Careless')
plt.scatter(x[y_kmeans ==4,0], x[y_kmeans ==4, 1], s = 100, c = 'violet', label='Sensible')
plt.title("Cluster of Clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()