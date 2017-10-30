import os, sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

# Global variables
SCRIPT_DIR = ""

def main():
	# Just set width of output when printing dataframe (Fit my screen)
	pd.set_option('display.width', 170)

	# Where this script is
	global SCRIPT_DIR
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Make csv data to dataframe
	data_dir = SCRIPT_DIR + "/../../input/clean/crime_counts_CLEANED.csv"
	df = pd.read_csv(data_dir)

	# Print sample data in dataframe
	print("========== Sample data ==========")
	print(df.iloc[:10])
	print()

	# Get rid of index, area_id: These features should not be used for using LOF because they are unique
	# Get rid of area_type: it is same for all rows
	train_df = df.drop(['area_id', 'area_type'], axis=1)
	train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]	# Remove Unnamed columns if there is

	# Categories to numeric
	train_df = pd.get_dummies(train_df)

	print("========== Sample train data ==========")
	print(train_df.iloc[:10])
	print()

	# Normalization to make it easier to illustrate
	min_max_scaler = preprocessing.MinMaxScaler()
	train_data = min_max_scaler.fit_transform(train_df)

	# Dataframe after get rid of some columns and numerize any category columns
	print("========== Normalized train data ==========")
	print(train_data[:10])
	print()

	# Try different k
	for k in [5, 10, 20]:
		# Fit the model
		clf = LocalOutlierFactor(n_neighbors=k)
		y_pred = clf.fit_predict(train_data)

		# Check that dataframe and array give the same result
		y_df_pred = clf.fit_predict(pd.DataFrame(train_data))
		for i, y in enumerate(y_pred):
			if y != y_df_pred[i]:
				print("Not same at", i)
				sys.exit()

		# Count number of outliers
		outlier_number = 0
		for y in y_pred:
			if y == -1:
				outlier_number += 1

		# Print sample prediction results
		print("========== Prediction results with k=%i ==========" % k)
		print("Total:", len(y_pred))
		print("Number of outliers:", outlier_number)
		print("Number of non-outliers:", (len(y_pred)-outlier_number))
		print("Oulier result:", y_pred)
		print()


if __name__ == "__main__":
	main()