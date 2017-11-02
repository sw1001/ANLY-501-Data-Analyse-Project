import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Correlations

Author: Shaobo Wang

Identify three (3) quantitative variables from either data set. Find the correlation between
all the pairs of these quantity variables. Include a table of the output in your report, and
explain your findings – what does this indicate about your data? Use scatter plots to
display the results. Ideally, create a set of scatter plot subplots.

'''


class Correlation:
    # data members
    file_name = ''
    data_frame = pd.DataFrame()

    # constructor
    def __init__(self, options):
        self.file_name = options[1]
        self.set_input_data()

    # read data from csv
    def set_input_data(self):
        self.data_frame = pd.read_csv(self.file_name)

    # get input data
    def get_input_data(self):
        return self.data_frame

    # correlation
    def corr(self):
        print(self.data_frame.corr(method='pearson'))
        # For zillow data, pick bathrooms, finishedSqFt and mount
        if self.file_name == 'Zillow_Cleaned.csv':
            plt.figure()
            plt.suptitle('Scatter plots of correlations in Zillow data')
            plt.subplot(1, 3, 1)
            plt.scatter(self.data_frame['bathrooms'], self.data_frame['finishedSqFt'])
            plt.xlabel("bathrooms")
            plt.ylabel("finishedSqFt")
            plt.subplot(1, 3, 2)
            plt.scatter(self.data_frame['bathrooms'], self.data_frame['amount'])
            plt.xlabel("bathrooms")
            plt.ylabel("amount")
            plt.subplot(1, 3, 3)
            plt.scatter(self.data_frame['finishedSqFt'], self.data_frame['amount'])
            plt.xlabel("finishedSqFt")
            plt.ylabel("amount")
            plt.show()

