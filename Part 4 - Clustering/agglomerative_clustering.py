# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:54:49 2023

@author: miwo
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]]

# Using dendograms finding the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()