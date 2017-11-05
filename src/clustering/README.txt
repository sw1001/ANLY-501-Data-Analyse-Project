This folder contains the following 3 files:

1) dbscan_Clustering.py: This program will perform DBSCAN on "crime_counts_CLEANED.csv" and will 
			 cluster the data for "States" and "All Crimes". In addition, it will 
			 calculate the silhouette ratio. 


2) kmeansClustering.py: This program will perform the kmeans clustering on "Zillow_Cleaned.csv" and will cluster
			the data for attributes ranging from "bathrooms" to "States".
			It will also display the silhouette score. Since the data has high dimensionality,
			it will perform a PCA projection into 2D.


3) HC.py: This program will perform Hierarchical Clustering on "graduation_rates_CLEANED.csv" and will 
	  cluster the data with attributes ranging from "percent_associates_degree"
	  to "State". In addition, it will also calculate the silhouette ratio.