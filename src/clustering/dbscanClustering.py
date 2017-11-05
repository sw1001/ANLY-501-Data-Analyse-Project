# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:18:42 2017

@author: Armaan Khullar
"""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from sklearn import cluster
import pylab
import sys
from pprint import pprint

def dbscan():
    
    plt.style.use("ggplot")
    
    myData = pd.read_csv("crime_counts_CLEANED.csv")
    lb = LabelEncoder()
    myData["State"] = lb.fit_transform(myData["State"])
    myData["City"] = lb.fit_transform(myData["City"])
        
    X = myData.iloc[:, [15,5, 6]].values
    #X = myData[["State", "All Crimes"]]  
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(X)
    #normalizedDataFrame = pd.DataFrame(x_scaled)
    
    #X, label = make_moons(n_samples=200, noise=0.1, random_state=19)
    print(X[:5,])
    
    model = DBSCAN(eps=.8, min_samples=12).fit(X) 
    print(model)  
    
    model.labels_
    model.core_sample_indices_
    
    #Plot the clusters in feature space.
    fig, ax = plt.subplots(figsize=(10, 8))
    sctr = ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=140, alpha=0.9, cmap=plt.cm.Set1)
    #fig.show()
    fig.savefig("DBSCAN.png")
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(X, model.labels_) #Get the silhouette score
    print("Silhouette avg:", str(silhouette_avg))
    
    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
            
    plot_columns = pca2D.fit_transform(X)
            
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model.labels_)
    #plt.show() #Display the plot
    plt.savefig("pca_dbscan.png")

dbscan()