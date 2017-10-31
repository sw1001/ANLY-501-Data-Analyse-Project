import os, sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor


def LOF(
		data_file_path,
		k_list=[5, 20, 50]):
	"""
	Use Local Outlier Facter algorithm to find outliers on the specific file
	:param data_file_path: The specific file path for input data
	:param k_list: (Optional) The list of neighbor numbers for LOF algorithm, default is k=5,20,50
	:return: None
	"""

	# Make csv data to dataframe
	df = pd.read_csv(data_file_path, encoding='latin-1')

	# Print sample data in dataframe
	print("========== Sample data ==========")
	print(df.iloc[:10])
	print()

	train_df = df

	# ===== Preprocessing ===== #

	# - Zillow file
	# 
	# Remove zpid column (Zillow ID)
	if 'zpid' in list(train_df):
		train_df = train_df.drop(['zpid'], axis=1)

	# Remove latitude and longitude columns
	# if 'latitude' in list(train_df):
	# 	train_df = train_df.drop(['latitude'], axis=1)
	# if 'longitude' in list(train_df):
	# 	train_df = train_df.drop(['longitude'], axis=1)

	# Remove countryid column because all are same
	if 'countryid' in list(train_df) and len(set(train_df['countryid'])) == 1:
		train_df = train_df.drop(['countryid'], axis=1)

	# Convert some columns to be categorical such as cityid, countryid, zipcpde
	if 'cityid' in list(train_df):
		train_df['cityid'] = train_df['cityid'].apply(lambda x: str(x)+"_categorized")
	if 'zipcpde' in list(train_df):
		train_df['zipcpde'] = train_df['zipcpde'].apply(lambda x: str(x)+"_categorized")

	# - Other files
	# 
	# Get rid of area_type, area_id: These features should not be used for using LOF because they are unique
	if 'area_id' in list(train_df) and 'area_type' in list(train_df):
		train_df = train_df.drop(['area_id', 'area_type'], axis=1)

	# Combine City and State to be one column
	# Or remove them because they are unique (or almost unique in some files)
	if 'City' in list(train_df) and 'State' in list(train_df):
		# train_df["CityState"] = train_df[['City', 'State']].apply(lambda x: ''.join(x), axis=1)
		train_df = train_df.drop(['City', 'State'], axis=1)

	# - All files
	# 
	# Get rid of area_type: it is same for all rows
	train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]	# Remove Unnamed columns if there is

	# Do one hot to convert categorical data to numeric data
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

	# ===== Local Outlier Factors ===== #
	# 
	# Set a file for outlier summary
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
	data_file_name = data_file_path.split('/')[len(data_file_path.split('/'))-1]
	summary_file_name = SCRIPT_DIR + '/' + data_file_name + '_outlier_summary.csv'
	f = open(summary_file_name, 'w')

	# Print header
	header = "k-neighbors,total, outliers, non-outliers, % outliers"
	f.write(header+'\n')

	# Try LOF with different k (default = 5, 20, 50)
	result_list = {}
	for k in k_list:
		# Fit the model
		clf = LocalOutlierFactor(n_neighbors=k)
		y_pred = clf.fit_predict(train_data)
		result_list["k="+str(k)] = y_pred

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
		print("Percentage of outliers:", "{0:.2f}%".format(outlier_number/(len(y_pred)-outlier_number)*100))
		print("Oulier result:", y_pred)
		print()

		# Write summary to file
		line = ",".join([str(k), str(len(y_pred)), str(outlier_number), str(len(y_pred)-outlier_number), "{0:.2f}%".format(outlier_number/(len(y_pred)-outlier_number)*100)])
		f.write(line+'\n')
	f.close()

	# Set another file for outlier results
	result_file_name = SCRIPT_DIR + '/' + data_file_name + '_outlier_results.csv'
	pd.DataFrame(result_list).to_csv(result_file_name, index=False)


def main():
	"""
	Run Local Outlier Factor algorithm to find outliers
	Uncomment to activate that file path since we have many cleaned files
	"""

	# Just set width of output when printing dataframe (Fit my screen)
	pd.set_option('display.width', 170)

	# Where this script is
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Select data file to apply LOF or loop for do many files
	data_file_paths = [None]*6
	data_file_paths[0] = SCRIPT_DIR + "/../../input/clean/Zillow_Cleaned.csv"			# 11.11
	data_file_paths[1] = SCRIPT_DIR + "/../../input/clean/crime_counts_CLEANED.csv"		# 11.16%
	data_file_paths[2] = SCRIPT_DIR + "/../../input/clean/crime_rates_CLEANED.csv"		# 11.11%
	data_file_paths[3] = SCRIPT_DIR + "/../../input/clean/earning_info_CLEANED.csv"		# 11.14%
	data_file_paths[4] = SCRIPT_DIR + "/../../input/clean/gdp_info_CLEANED.csv"			# 11.11%
	data_file_paths[5] = SCRIPT_DIR + "/../../input/clean/graduation_rates_CLEANED.csv"	# 11.42%
	
	# Local Outlier Factor
	for i in range(len(data_file_paths)):
		if data_file_paths[i] != None:
			print("===============================================")
			print("========== Data file path number %i: ==========" % i)
			print("===============================================\n")
			LOF(data_file_paths[i])
			print()


if __name__ == "__main__":
	main()