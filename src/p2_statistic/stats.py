import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Basic Statistical Analysis and data cleaning insight

Author: Shaobo Wang

Determine the mean (mode if categorical), median, and standard deviation of at least 10
attributes in your data sets. Use Python to generate these results and use the project
report to show and explain each.

In the last assignment you took several steps to clean your data. Here you need to check
to make sure that the cleaning decisions you made make sense for the analysis you will
do in this assignment. To do this, consider your raw data and consider your current
cleaned data. Next, do the following:
o Identify any attributes that may contain outliers. If you did not deal with outliers in
Project 1, do this now by writing Python3 code to locate and potentially clean
outliers. In your report, note the attributes that contained potential outliers (you
do not have to list all the outliers themselves)
o Explain how you detected the outliers, and how you made the decision to keep or
remove them.
o From the cleaning phase of Project 1, also discuss which attributes had missing
values and explain your strategy for handling them.
o If you find that you data needs to be further cleaned or differently cleaned based
on analyses, include explanations here. Be specific about what you did and why.

'''


class Stats:
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

    # get statistics
    def get_stats(self):
        print("Basic stats descriptions: ")
        print(self.data_frame.describe())
        # self.data_frame.column.agg(['count', 'min', 'max', 'mean'])
