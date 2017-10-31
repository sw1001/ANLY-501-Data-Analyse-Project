import numpy as np
import pandas as pd
import sys, os


def association_rules(
		data_file_path,
		k_list=[5, 20, 50]):
	


def main():
	"""
	Run Apriori algorithm for association rules
	Uncomment to activate that file path since we have many cleaned files
	Note that, the algorithm run very slow, so we will use only sample of dataset
	"""

	# Just set width of output when printing dataframe (Fit my screen)
	pd.set_option('display.width', 170)

	# Where this script is
	SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Select data file to apply LOF or loop for do many files
	# Uncomment to activate file paths
	data_file_paths = [None]*6
	# data_file_paths[0] = SCRIPT_DIR + "/../../input/clean/Zillow_Cleaned.csv"
	# data_file_paths[1] = SCRIPT_DIR + "/../../input/clean/crime_counts_CLEANED.csv"
	# data_file_paths[2] = SCRIPT_DIR + "/../../input/clean/crime_rates_CLEANED.csv"
	# data_file_paths[3] = SCRIPT_DIR + "/../../input/clean/earning_info_CLEANED.csv"
	# data_file_paths[4] = SCRIPT_DIR + "/../../input/clean/gdp_info_CLEANED.csv"
	data_file_paths[5] = SCRIPT_DIR + "/../../input/clean/graduation_rates_CLEANED.csv"
	
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