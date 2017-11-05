import sys, os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from statistics import stdev
from apyori import apriori


def binning(dataset, column_name):
	"""
	Binning to 3 categories, 'Mean': mean+0.5SD > data > mean-0.5SD, 'High': data > mean+0.5SD, 'Low': data < mean-0.5SD
	:param dataset: Dataset used for binning
	:return: None
	"""

	dataset = list(dataset)
	mean = sum(dataset)/len(dataset)
	# std = stdev(dataset)	# Overflow
	std = np.std(dataset)

	binned_dataset = []
	for i, d in enumerate(dataset):
		if d >= mean-(std/2) and d <= mean+(std/2):
			dataset[i] = column_name+"_Mean"
		elif d > mean+(std/2):
			dataset[i] = column_name+"_High"
		else:
			dataset[i] = column_name+"_Low"

	return dataset


def association_rules(
		data_file_path,
		min_sups=[0.2, 0.5, 0.8],
		min_confs=[0.2, 0.5, 0.8],
		min_items=3,
		sample_percentage=None,
		sample_number=None,
		print_rules=False):
	"""
	Find association rules using Apriori algorithm
	:param data_file_path: The specific file path for input dataset
	:param min_sups: (Optional) List of minimum support levels for Apriori algorithm
	:param min_confs: (Optional) List of minimum confident levels for Apriori algorithm
	:param min_items: (Optional) Minimum number of items in association rules
	:param sample_percentage: (Optional) Size of sample data in percenrage
	:param print_rules: (Optional) print rules or not
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

	# Remove latitude and longitude columns (too unique)
	if 'latitude' in list(train_df):
		train_df = train_df.drop(['latitude'], axis=1)
	if 'longitude' in list(train_df):
		train_df = train_df.drop(['longitude'], axis=1)

	# Remove countryid column because all are same
	if 'countryid' in list(train_df) and len(set(train_df['countryid'])) == 1:
		train_df = train_df.drop(['countryid'], axis=1)

	# Convert some columns to be categorical such as cityid, countryid, zipcpde
	if 'cityid' in list(train_df):
		train_df['cityid'] = train_df['cityid'].apply(lambda x: "cityid"+str(x)+"_categorized")
	if 'zipcpde' in list(train_df):
		train_df['zipcpde'] = train_df['zipcpde'].apply(lambda x: "zipcpde"+str(x)+"_categorized")

	# - Other files
	# 
	# Get rid of area_type, area_id: These features should not be used because they are unique (area_id) and same (area_type)
	if 'area_id' in list(train_df) and 'area_type' in list(train_df):
		train_df = train_df.drop(['area_id', 'area_type'], axis=1)

	# - All files
	# 
	# Get rid of area_type: it is same for all rows
	train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]	# Remove Unnamed columns if there is

	# If all data in a column are same, we should not use that column for association rules
	del_cols = []
	for idx, column in enumerate(train_df.columns):
		if len(set(train_df[column])) <= 1:
			del_cols.append(column)
	train_df = train_df.drop(del_cols, axis=1)

	# If data are numeric and different more than 13 elements
	# We have to ensure than we didn't change numeric data like arecondition_type (0, 1), etc.
	# Change numeric data columns to categorical data columns
	for idx, column_dtype in enumerate(train_df.dtypes):
		if column_dtype != "object":
			if len(set(train_df[train_df.columns[idx]])) > 13:
				# Binning to 3 categories
				train_df[train_df.columns[idx]] = binning(train_df[train_df.columns[idx]], train_df.columns[idx])
			else:
				# Just convert numeric to categorical
				train_df[train_df.columns[idx]] = train_df[train_df.columns[idx]].apply(lambda x: str(train_df.columns[idx])+"_"+str(x))

	
	# Get sample data (default is 300 sample data)
	if sample_percentage != None and sample_number == None:
		sample_percentage = min(sample_percentage, 1.0)		# Not over the data size
		train_df = train_df.sample(frac=sample_percentage, replace=False)
	elif sample_number != None and sample_percentage == None:
		sample_number = min(sample_number, len(train_df.index))		# Not over the data size
		train_df = train_df.sample(n=sample_number, replace=False)
	elif sample_number != None and sample_percentage != None:
		sample_number = min(sample_number, int(sample_percentage*len(train_df.index)))
		sample_number = min(sample_number, len(train_df.index))		# Not over the data size
		train_df = train_df.sample(n=sample_number, replace=False)


	print("========== Sample train data ==========")
	print(train_df.iloc[:10])
	print()

	# Log file
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))	# Where this script is
	data_file_name = data_file_path.split("/")[len(data_file_path.split("/"))-1]
	f = open(SCRIPT_DIR+'/'+data_file_name+'_association_rules_log.txt', 'w')

	# Data information
	print("========== Data Information ==========")
	f.write("========== Data Information ==========\n")
	print("Original data: %i rows" % len(df.index))
	f.write("Original data: %i rows\n" % len(df.index))
	print("Sample data: %i rows" % len(train_df.index))
	f.write("Sample data: %i rows\n\n" % len(train_df.index))
	print()
	
	# Apriori using different minimum supports and minimum confidences
	transactions = train_df.values.tolist()
	for min_sup in min_sups:
		for min_conf in min_confs:
			print("========== Compute: min_sup: %.2f & min_conf: %.2f ==========" % (min_sup, min_conf))
			f.write("========== Compute: min_sup: %.2f & min_conf: %.2f ==========\n" % (min_sup, min_conf))

			# Apriori algorithm
			result = list(apriori(transactions, min_support=min_sup, min_confidence=min_conf))

			# Results
			rule_set_number = 0
			max_support = -1
			most_frequent_itemset = None
			for r in result:
			    if len(r.items) >= min_items:
			    	if print_rules:
				    	print("==========")
				    	f.write("==========\n")
				    	print(">>> Itemset:", str(set(r.items)))
				    	f.write(" ".join([">>> Itemset:", str(set(r.items)), '\n']))
				    	print(">>> Support:", str(r.support), '\n')
				    	f.write(" ".join([">>> Support:", str(r.support), '\n\n']))

				    	# Update most frequent itemset
				    	if max_support < r.support:
				    		max_support = r.support
				    		most_frequent_itemset = set(r.items)

				    	# Print association rules
				    	for idx, rule in enumerate(r.ordered_statistics):
				    		print(">>> Rule", str(idx+1), ":", str(set(rule.items_base)), "--->", str(set(rule.items_add)))
				    		f.write(" ".join([">>> Rule", str(idx+1), ":", str(set(rule.items_base)), "--->", str(set(rule.items_add)), '\n']))
				    		print("Confidence: " + str(rule.confidence) + '\n')
				    		f.write("Confidence: " + str(rule.confidence) + '\n\n')
			    	rule_set_number += 1
			print("========== Summary: min_sup: %.2f & min_conf: %.2f ==========" % (min_sup, min_conf))
			f.write("========== Summary:  min_sup: %.2f & min_conf: %.2f ==========\n" % (min_sup, min_conf))
			print("Rule Set Number: "+str(rule_set_number)+'\n')
			f.write("Rule Set Number: "+str(rule_set_number)+'\n')
			print("Most frequent itemset: " + str(most_frequent_itemset) + ' with support ' + str(max_support))
			f.write("Most frequent itemset: " + str(most_frequent_itemset) + ' with support ' + str(max_support) + '\n\n')

	f.close()


def main():
	"""
	Run Apriori algorithm association rules
	Uncomment to activate that file path since we have many cleaned files
	"""

	# Just set width of output when printing dataframe (Fit my screen)
	pd.set_option('display.width', 170)

	# Where this script is
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Select data file to apply LOF or loop for do many files
	# Uncomment to activate file paths
	data_file_paths = [None]*6
	data_file_paths[0] = SCRIPT_DIR + "/../../input/clean/Zillow_Cleaned.csv"
	data_file_paths[1] = SCRIPT_DIR + "/../../input/clean/crime_counts_CLEANED.csv"
	data_file_paths[2] = SCRIPT_DIR + "/../../input/clean/crime_rates_CLEANED.csv"
	data_file_paths[3] = SCRIPT_DIR + "/../../input/clean/earning_info_CLEANED.csv"
	data_file_paths[4] = SCRIPT_DIR + "/../../input/clean/gdp_info_CLEANED.csv"
	data_file_paths[5] = SCRIPT_DIR + "/../../input/clean/graduation_rates_CLEANED.csv"
	
	# Local Outlier Factor
	for i in range(len(data_file_paths)):
		if data_file_paths[i] != None:
			print("===============================================")
			print("========== Data file path number %i: ==========" % i)
			print("===============================================\n")
			
			# To avoid making my computer crash, we will use only small sample set
			# Set sample data to 300 samples
			association_rules(data_file_paths[i], sample_percentage=None, sample_number=3000, print_rules=True)


if __name__ == "__main__":
	main()