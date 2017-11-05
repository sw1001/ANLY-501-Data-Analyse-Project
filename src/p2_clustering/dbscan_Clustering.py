# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 01:27:25 2017

@author: Armaan Khullar

This program will perform DBSCAN on "crime_counts_CLEANED.csv" and will 
cluster the data for "States" and "All Crimes". In addition, it will 
calculate the silhouette ratio.

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
    
    #plt.style.use("ggplot")
    
    X = pd.read_csv("/input/clean/crime_counts_CLEANED.csv")
    
    #Since "City" and "States" are strings, we convert them to numeric representations
    #using LabelEncoder().
    lb = LabelEncoder()
    X["State"] = lb.fit_transform(X["State"])
    X["City"] = lb.fit_transform(X["City"])
        
    X = X.iloc[:, [15,5]].values #Storing only the columns "State" and "All Crimes"
    
    X = StandardScaler().fit_transform(X) #Fit the data, then transform it.
    y_pred = DBSCAN(eps=0.8, min_samples=30).fit_predict(X) #Use DBSCAN on the data
    
    plt.scatter(X[:,0], X[:,1], c=y_pred) #Setup the scatterplot for the clusters
    plt.show()                             #Display the clusters
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(X, y_pred) #Get the silhouette score
    print("Silhouette avg:", str(silhouette_avg))
    

if __name__ == "__main__":
    dbscan()
    