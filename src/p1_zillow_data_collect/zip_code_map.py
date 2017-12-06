"""
File: zip_code_map.py
Author: Zhuoran Wu
Email: zw118@georgetown.edu
"""

import pandas as pd
import math
import googlemaps

data_filename = 'Zillow_House_Final.csv'
data_filename_out = 'Zillow_House_Final_Out.csv'
GoogleAPIKey = 'AIzaSyCSRNt6-gf8XiTCUKP2udaW6WJZWh7ykGE'


def transfer_zipcode_to_city_state(dataframe):
    """
    Give an dataframe with zpid, zip code, transfer to City and State via Google Map API.
    :param dataframe: The Dataframe with zpid and zipcode
    :return: the updated dataframe.
    """
    dataframe['city'] = ""
    dataframe['county'] = ""
    dataframe['state'] = ""

    gmaps = googlemaps.Client(key=GoogleAPIKey)

    for index, row in dataframe.iterrows():
        lat = row[6]
        long = row[7]
        lat /= 10e5
        long /= 10e5

        result = gmaps.reverse_geocode((lat, long))

        if len(result) == 0:
            continue

        if len(result[0]['address_components']) <= 7:
            continue

        # print(result)
        print(len(result[0]['address_components']))

        city = result[0]['address_components'][3]['short_name']
        county = result[0]['address_components'][4]['short_name']
        state = result[0]['address_components'][5]['short_name']
        zip = result[0]['address_components'][7]['short_name']

        dataframe.loc[dataframe['zpid'] == row[0], 'city'] = city
        dataframe.loc[dataframe['zpid'] == row[0], 'county'] = county
        dataframe.loc[dataframe['zpid'] == row[0], 'state'] = state
        dataframe.loc[dataframe['zpid'] == row[0], 'zipcode'] = zip

    return dataframe


def main():
    dataframe = pd.read_csv('../../input/raw/' + data_filename)
    dataframe = transfer_zipcode_to_city_state(dataframe)
    dataframe.to_csv('../../input/' + data_filename_out)


if __name__ == "__main__":
    main()
