# -*- coding: utf-8 -*-
"""customer segmentation using kmeans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eNAB52XfswATByGEBkuxTM_g2cCrf6Cf

importing the libaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import  KMeans

"""Data Collection And Analysis

"""

import os
file_path = r"C:/Users/nitro/OneDrive/Desktop/Mall_Customers.csv"
print(os.path.exists(file_path))  # This will return True if the file exists, False if not

data = pd.read_csv('/content/Mall_Customers.csv')

data.head(5)

"""Getting some Info about the dataset"""

data.shape

data.info()

data.describe()

data.isnull().sum()

"""choosing the annual income column and spending score column"""

x=data.iloc[:,[3,4]].values

print(x)

"""Choosing the number of clusters

(WCSS - within clusters sum of squares )
is the technique that we will use to find the number of clusters
"""

# FINDING THE WCSS VALUES

wcss=[]

for i in range(1,11):
  kmeans= KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(x)

  wcss.append(kmeans.inertia_)
  # inertia method returns wcss for that model

#plot the elbow graph

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""The Optimum numbers of cluster will be 5"""

#Training the k=means Clustering Model

kmeans=KMeans(n_clusters=5 , init = 'k-means++' ,random_state=0 )

#return a label for each data point based on thier cluster

y=kmeans.fit_predict(x)
print(y)

"""Visualize the Clusters

"""

#plotting all thier clusters and centroids

plt.figure(figsize=(10,8))
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='blue',label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='red',label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='yellow',label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='orange',label='Cluster 5')

#plotting the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')

plt.title('Customer Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

