import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt


'''
Binning a numeric variable

Author: Shaobo Wang

For at least one of the numeric variables in one of the data sets, write code to bin the
data. This will create a new column. Use the binning strategy that is most intuitive for
your data. Explain your decision. Include why you chose to bin the specific attribute
selected, the binning method used, and why that method makes sense for your data.

'''

class Bin:
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

    # bin and add a new column
    def bin(self):
        i = 0
        for column in self.data_frame.columns:
            if self.data_frame.dtypes[i] == 'int64':
                # bin the data
                bins = pd.cut(self.data_frame[column], 10, retbins=True, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                print('Bins for ' + str(column) + ' is: ')
                print(bins[0])
                bins[0].to_csv(str(column) + '_bin.csv')
            i += 1
