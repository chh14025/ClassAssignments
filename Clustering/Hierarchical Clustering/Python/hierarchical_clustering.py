#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:38:11 2020

@author: s.p.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)


plt.scatter(x[y_hc ==0,0], x[y_hc ==0, 1], s = 100, c = 'red', label='Careful')
plt.scatter(x[y_hc ==1,0], x[y_hc ==1, 1], s = 100, c = 'blue', label='Standard')
plt.scatter(x[y_hc ==2,0], x[y_hc ==2, 1], s = 100, c = 'green', label='Target')
plt.scatter(x[y_hc ==3,0], x[y_hc ==3, 1], s = 100, c = 'cyan', label='Careless')
plt.scatter(x[y_hc ==4,0], x[y_hc ==4, 1], s = 100, c = 'violet', label='Sensible')
plt.title("Cluster of Clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()