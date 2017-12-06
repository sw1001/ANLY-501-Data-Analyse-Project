import pandas as pd

zillow = pd.read_csv('../../input/raw/Zillow_House_Final.csv')

filter = zillow[zillow['cityid'].str.contains("([A-z]+)",  na=False)]

filter.to_csv('../../input/raw/Zillow_House_Final.csv')
