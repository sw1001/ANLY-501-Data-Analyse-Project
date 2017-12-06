import pandas as pd
from scipy.spatial import distance

zillow = pd.read_csv('Zillow_Cleaned_3.csv')


def calculate(row):
    city = row['cityid']
    zpid = row['zpid']
    x = row['latitude']
    y = row['longitude']
    vector1 = [x, y]

    samecity = zillow.loc[zillow['cityid'] == city]
    count = 0
    distall = 0.0
    amountall = 0.0

    for idx, row1 in samecity.iterrows():

        x1 = row1['latitude']
        y2 = row1['longitude']
        vector2 = [x1, y2]
        dist = distance.euclidean(vector1, vector2)
        if dist < 0.005:
            distall = distall + dist
            amountall = amountall + row1['amount']
            count = count + 1

    avg = 0.0
    if count > 0:
        avg = amountall / count
    else:
        avg = amountall / 1

    print(str(avg) + "-" + str(count))
    zillow.loc[zillow['zpid'] == zpid, 'neighborhood_avg_amount'] = avg

zillow['neighborhood_avg_amount'] = 0.0

zillow.apply(calculate, axis=1)

zillow.to_csv('Zillow_Cleaned_4.csv', index=False)
