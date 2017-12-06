import plotly
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.figure_factory as ff
import numpy as np

'''
this is the python code to create a series of interactive statistical charts using plotly.
'''


# credentials for using plotly online mode
plotly.tools.set_credentials_file(username='shaun2525', api_key='Nd0vfSKIHN4nNVZlbLOk')

# Setup data frame
data_frame = pd.read_csv('../../input/raw/Zillow_House_Final_40000.csv')

###
# data clean
###
data_frame = data_frame.dropna(axis=0, how='any')

data_frame['amount'] = data_frame['amount'].astype('int64')
data_frame['finishedSqFt'] = data_frame['finishedSqFt'].astype('int64')
data_frame['lotsizeSqFt'] = data_frame['lotsizeSqFt'].astype('int64')

print(data_frame['amount'].dtype)
print(data_frame['finishedSqFt'].dtype)
print(data_frame['lotsizeSqFt'].dtype)

data_frame = data_frame.loc[(data_frame['amount'] < 2000000) &
                            (data_frame['finishedSqFt'] < 50000) &
                            (data_frame['amount'] > 30000)]

###
# box plot
###
# Create a trace for a box plot
trace = go.Box(y=data_frame['amount'],
               name='amount',
               boxpoints='all')

# Assign it to an iterable object named myData
my_data = [trace]

# Add axes and title
my_layout = go.Layout(title='Box plot for house price of Zillow data')

# Setup figure
my_figure = go.Figure(data=my_data, layout=my_layout)

# Create the box plot
py.plot(my_figure, filename='box_zillow_amount')

###
# 2d density plot
###
# t = np.linspace(-1, 1.2, 2000)
# x = (t**3) + (0.3 * np.random.randn(2000))
# y = (t**6) + (0.3 * np.random.randn(2000))

x = data_frame['finishedSqFt']
y = data_frame['amount']
'''
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]

fig = ff.create_2d_density(
    x, y, colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=3
)
'''
trace1 = go.Scatter(
    x=x, y=y, mode='markers', name='points',
    marker=dict(color='#D56073', size=3, opacity=0.4)
)
# trace2 = go.Histogram2dcontour(
#     x=x, y=y, name='density', ncontours=20,
#     colorscale='Hot', reversescale=True, showscale=False
# )
trace3 = go.Histogram(
    x=x, name='x density',
    marker=dict(color='rgb(255, 237, 222)'),
    yaxis='y2'
)
trace4 = go.Histogram(
    y=y, name='y density', marker=dict(color='rgb(255, 237, 222)'),
    xaxis='x2'
)
data = [trace1, trace3, trace4]

layout = go.Layout(
    title='Price vs Area',
    showlegend=False,
    autosize=False,
    width=600,
    height=550,
    xaxis=dict(
        title='sqft',
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='dollar',
        domain=[0, 0.85],
        showgrid=False,
        zeroline=False
    ),
    margin=dict(
        t=50
    ),
    hovermode='closest',
    bargap=0,
    xaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    ),
    yaxis2=dict(
        domain=[0.85, 1],
        showgrid=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='histogram_subplots')

###
# basic histogram
###
data = [go.Histogram(x=data_frame['amount'])]
layout = go.Layout(title='House price histogram', xaxis=dict(title='amount'))
figure = go.Figure(data=data, layout=layout)

py.plot(figure, filename='histogram_amount_zillow')

#####################################

