import plotly
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go

'''
this is the python code to create a series of interactive statistical charts using plotly.
'''

# credentials for using plotly online mode
plotly.tools.set_credentials_file(username='shaun2525', api_key='qqTkmMmqX2fr9H6QHGBp')

# Setup data frame
dataFrame = pd.read_csv('2012_Workplace_Fatalities_fixed_final.csv')
