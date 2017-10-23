"""
File: get_zillow_data.py
Author: Zhuoran Wu
Email: zw118@georgetown.edu
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re

house_info_filename = 'Zillow_House_Info.csv'
house_detail_filename = 'Zillow_House_Detail.csv'
house_updated_filename = 'Zillow_House_Update.csv'
house_zpidList_filename = 'Zillow_House_zpid_list.csv'


def get_zestimate_zpi(
        zwisd,
        zpid):
    """
    Get House Info via GetZestimateAPI
    :param zwisd: The API KEY for API
    :param zpid: The id for single house
    :return: Response from Request
    """
    get_zestimate_zpi_baseurl = 'http://www.zillow.com/webservice/GetZestimate.htm'
    get_zestimate_zpi_urlpost = {'zws-id': str(zwisd),
                                 'zpid': str(zpid)}
    response = requests.get(get_zestimate_zpi_baseurl, get_zestimate_zpi_urlpost)
    return response


def get_deepsearch_result_api(
        zwisd,
        address,
        citystatezip):
    """
    Get House Detail Info via GetDeepSearch-Results API
    :param zwisd: The API KEY for API
    :param address: The address for a single house
    :param citystatezip: City and State
    :return: Response from Request
    """
    get_deepsearch_result_zpi_baseurl = 'http://www.zillow.com/webservice/GetDeepSearchResults.htm'
    get_deepsearch_result_zpi_urlpost = {
        'zws-id' : str(zwisd),
        'address' : address,
        'citystatezip' : citystatezip}
    response = requests.get(get_deepsearch_result_zpi_baseurl, get_deepsearch_result_zpi_urlpost)
    return response


def get_updated_property_details_api(
        zwisd,
        zpid):
    """
    Get House Detail Info via Updated Property Details API
    :param zwisd: The API KEY for API
    :param zpid: The id for single house
    :return: Response from Request
    """
    get_updated_property_details_zpi_baseurl = 'http://www.zillow.com/webservice/GetUpdatedPropertyDetails.htm'
    get_updated_property_details_zpi_urlpost = {
        'zws-id': str(zwisd),
        'zpid': str(zpid)}
    response = requests.get(get_updated_property_details_zpi_baseurl, get_updated_property_details_zpi_urlpost)
    return response


def get_rows_from_zestimate_zpi(
        dataframe,
        row_start,
        row_end,
        zwisd):
    """
    Open a dataframe, and get the zestimate API from row_start to row_end with ZWISD.
    :param dataframe: The dataframe with zpid id
    :param row_start: The row we start to collect data.
    :param row_end: The row we end to collect data
    :param zwisd: The zwisd API KEY
    :return: the update dataframe
    """
    if len(dataframe) < row_end:
        print('Get Zestimage API row larger than dataframe lenght.')
        return

    for zpid in dataframe.iloc[row_start:row_end]['zpid']:
        # Get response in XML format
        response = get_zestimate_zpi(zwisd, zpid)
        # Remove namespace for better using.
        contents = re.sub('<!--(.*?)-->', '', str(response.text))
        contents = re.sub(':zestimate+.*xsd/Zestimate.xsd"', '', contents)
        contents = re.sub(':zestimate', '', contents)

        root = ET.fromstring(contents)

        if root[1][1].text != '0':
            print('Get Zestimate API cannot get data with zpid:' + str(zpid))
            continue

        if len(root[1]) > 2:
            # The call are approaching the limit per day.
            print('Get Zestimate API are approaching call limit today. Call Terminate.')
            return

        # root[1][1].text       message code
        # root[2][0].text       zpid
        # root[2][2][0].text    street
        # root[2][2][1].text    zipcode
        # root[2][2][2].text    city
        # root[2][2][3].text    state
        # root[2][2][4].text    latitude
        # root[2][2][5].text    longitude
        # root[2][3][0].text    amount
        # root[2][3][1].text    last-updated date
        # root[2][3][3].text    valueChange
        # root[2][5][0].text    zipcode_id
        # root[2][5][1].text    city_id
        # root[2][5][2].text    county_id
        # root[2][5][3].text    state_id

        dataframe.loc[dataframe['zpid'] == zpid, 'latitude'] = str(float(root[2][2][4].text) * 1000000)
        dataframe.loc[dataframe['zpid'] == zpid, 'longitude'] = str(float(root[2][2][5].text) * 1000000)
        dataframe.loc[dataframe['zpid'] == zpid, 'cityid'] = root[2][5][1].text
        dataframe.loc[dataframe['zpid'] == zpid, 'countyid'] = root[2][5][2].text
        dataframe.loc[dataframe['zpid'] == zpid, 'zipcode'] = root[2][5][0].text
        dataframe.loc[dataframe['zpid'] == zpid, 'amount'] = root[2][3][0].text
        dataframe.loc[dataframe['zpid'] == zpid, 'street'] = root[2][2][0].text
        dataframe.loc[dataframe['zpid'] == zpid, 'city'] = root[2][2][2].text
        dataframe.loc[dataframe['zpid'] == zpid, 'state'] = root[2][2][3].text

    # dataframe.to_csv('../input/' + HouseInfoFileName, index=False)
    return dataframe


def get_rows_from_deepsearch_result_api(
        dataframe,
        datainfo,
        row_start,
        row_end,
        zwisd):
    """
    Open a dataframe, and get the deep search result API from row_start to row_end with address and citystate.
    :param dataframe: The dataframe with zpid, address and citystate
    :param datainfo: The info data which contains street, city and state
    :param row_start: The row we start to collect data.
    :param row_end: The row we end to collect data
    :param zwisd: The zwisd API KEY
    :return: the update dataframe
    """
    if len(dataframe) < row_end:
        print('Get Deep Search Result API row larger than dataframe lenght.')
        return

    for zpid in dataframe.iloc[row_start:row_end]['zpid']:
        # Get response in XML format
        if datainfo.loc[datainfo['zpid'] == zpid]['street'].empty:
            continue

        response = get_deepsearch_result_api(
            zwisd,
            datainfo.loc[datainfo['zpid'] == zpid]['street'],
            datainfo.loc[datainfo['zpid'] == zpid]['city'] + ', ' + datainfo.loc[datainfo['zpid'] == zpid]['state'])
        # Remove namespace for better using.
        contents = re.sub('<!--(.*?)-->', '', str(response.text))
        contents = re.sub(':searchresults+.*xsd/SearchResults.xsd"', '', contents)
        contents = re.sub(':searchresults', '', contents)

        root = ET.fromstring(contents)

        if root[1][1].text != '0':
            print('Get Deep Search Result API cannot get data with zpid:' + str(zpid))
            continue

        if len(root[1]) > 2:
            # The call are approaching the limit per day.
            print('Get Deep Search Result API are approaching call limit today. Call Terminate.')
            return

        # root[1][1].text           message code
        # root[2][0][0][0].text     zpid
        # root[2][0][0][7].text     yearbuilt
        # root[2][0][0][8].text     lotSizeSqFt
        # root[2][0][0][9].text     finishedSqFt
        # root[2][0][0][10].text    bathrooms
        # root[2][0][0][11].text    bedrooms
        # root[2][0][0][2][4].text  latitude
        # root[2][0][0][2][5].text  longitude
        # root[2][0][0][14][0].text amount

        try:
            dataframe.loc[dataframe['zpid'] == zpid, 'bathrooms'] = root[2][0][0][10].text
            dataframe.loc[dataframe['zpid'] == zpid, 'bedrooms'] = root[2][0][0][11].text
            dataframe.loc[dataframe['zpid'] == zpid, 'finishedSqFt'] = root[2][0][0][9].text
            dataframe.loc[dataframe['zpid'] == zpid, 'lotsizeSqFt'] = root[2][0][0][8].text
            dataframe.loc[dataframe['zpid'] == zpid, 'latitude'] = str(float(root[2][0][0][2][4].text) * 1000000)
            dataframe.loc[dataframe['zpid'] == zpid, 'longitude'] = str(float(root[2][0][0][2][5].text) * 1000000)
            dataframe.loc[dataframe['zpid'] == zpid, 'yearbuilt'] = root[2][0][0][7].text
            dataframe.loc[dataframe['zpid'] == zpid, 'amount'] = root[2][0][0][14][0].text
        except IndexError:
            print('Get Deep Search Result API cannot get data with zpid:' + str(zpid))

    # dataframe.to_csv('../input/' + HouseInfoFileName, index=False)
    return dataframe


def get_rows_from_updated_property_details_api(
        dataframe,
        row_start,
        row_end,
        zwisd):
    """
    Open a dataframe, and get the updated property result API from row_start to row_end with zpid.
    :param dataframe: The dataframe with zpid, address and citystate
    :param row_start: The row we start to collect data.
    :param row_end: The row we end to collect data
    :param zwisd: The zwisd API KEY
    :return: the update dataframe
    """

    if len(dataframe) < row_end:
        print('Get Updated Property API row larger than dataframe lenght.')
        return

    for zpid in dataframe.iloc[row_start:row_end]['zpid']:
        # Get response in XML format
        response = get_updated_property_details_api(zwisd, zpid)
        # Remove namespace for better using.
        contents = re.sub('<!--(.*?)-->', '', str(response.text))
        contents = re.sub(':updatedPropertyDetails+.*XMLSchema-instance"', '', contents)
        contents = re.sub(':updatedPropertyDetails', '', contents)

        root = ET.fromstring(contents)

        if root[1][1].text != '0':
            print('Get Updated Property API cannot get data with zpid:' + str(zpid))
            continue

        if len(root[1]) > 2:
            # The call are approaching the limit per day.
            print('Get Updated Property API are approaching call limit today. Call Terminate.')
            return

        # root[1][1].text       message code
        # root[2][2][4].text    latitude
        # root[2][2][5].text    longitude
        # root[2][5][7].text    units
        # root[2][5][14].text   rooms
        # root[2][5][13].text   heatsystemtype
        # root[2][5][11].text   airconditiontype
        # root[2][5][8].text    buildingquality
        try:
            dataframe.loc[dataframe['zpid'] == zpid, 'latitude'] = str(float(root[2][2][4].text) * 1000000)
            dataframe.loc[dataframe['zpid'] == zpid, 'longitude'] = str(float(root[2][2][5].text) * 1000000)
            dataframe.loc[dataframe['zpid'] == zpid, 'units'] = root[2][5][7].text
            dataframe.loc[dataframe['zpid'] == zpid, 'rooms'] = root[2][5][14].text
            dataframe.loc[dataframe['zpid'] == zpid, 'heatsystemtype'] = root[2][5][13].text
            dataframe.loc[dataframe['zpid'] == zpid, 'airconditiontype'] = root[2][5][11].text
            dataframe.loc[dataframe['zpid'] == zpid, 'buildingquality'] = root[2][5][8].text
        except IndexError:
            print('Get Updated Property API cannot get data with zpid:' + str(zpid))

        # dataframe.to_csv('../input/' + HouseInfoFileName, index=False)

    return dataframe


def main():
    ZWSID1 = 'X1-ZWz1g08hy6pukr_3csw9'
    ZWSID2 = 'X1-ZWz1g0k86stlaj_1pt0f'
    ZWSID3 = 'X1-ZWz18zy8rmtc7f_1r7kw'
    ZWSID4 = 'X1-ZWz1bgfgbsq58r_1llb0'

    house_info_column = ['zpid',
                         'latitude',
                         'longitude',
                         'cityid',
                         'countyid',
                         'zipcode',
                         'amount',
                         'street',
                         'city',
                         'state']

    house_detail_column = ['zpid',
                           'bathrooms',
                           'bedrooms',
                           'fullbathrooms',
                           'finishedSqFt',
                           'lotsizeSqFt',
                           'latitude',
                           'longitude',
                           'yearbuilt',
                           'amount']

    house_update_column = ['zpid',
                           'rooms',
                           'units',
                           'airconditiontype',
                           'heatsystemtype,',
                           'buildingquality',
                           'pools']

    # Read Zpid List
    data_zpid_list = pd.read_csv('../../input/raw/' + house_zpidList_filename)
    data_info = data_zpid_list.reindex(columns=house_info_column)
    data_info[house_info_column[1:]] = ""
    data_deep = data_zpid_list.reindex(columns=house_detail_column)
    data_deep[house_detail_column[1:]] = ""
    data_update = data_zpid_list.reindex(columns=house_update_column)
    data_update[house_update_column[1:]] = ""

    # Get Info Data
    data_info_temp = get_rows_from_zestimate_zpi(data_info, 10, 20, ZWSID4)
    data_info = data_info.append(data_info_temp)
    # data_info_temp = get_rows_from_zestimate_zpi(data_info, 20, 30, ZWSID2)
    # data_info = data_info.append(data_info_temp)
    # data_info_temp = get_rows_from_zestimate_zpi(data_info, 30, 40, ZWSID3)
    # data_info = data_info.append(data_info_temp)
    # data_info_temp = get_rows_from_zestimate_zpi(data_info, 40, 50, ZWSID4)
    # data_info = data_info.append(data_info_temp)

    # Get Deep Data
    data_deep_temp = get_rows_from_deepsearch_result_api(data_deep, data_info, 10, 20, ZWSID4)
    data_deep = data_deep.append(data_deep_temp)
    # data_deep_temp = get_rows_from_deepsearch_result_api(data_deep, data_info, 20, 30, ZWSID2)
    # data_deep = data_deep.append(data_deep_temp)
    # data_deep_temp = get_rows_from_deepsearch_result_api(data_deep, data_info, 30, 40, ZWSID3)
    # data_deep = data_deep.append(data_deep_temp)
    # data_deep_temp = get_rows_from_deepsearch_result_api(data_deep, data_info, 40, 50, ZWSID4)
    # data_deep = data_deep.append(data_deep_temp)

    # Get update Date
    data_update_temp = get_rows_from_updated_property_details_api(data_update, 10, 20, ZWSID4)
    data_update = data_update.append(data_update_temp)
    # data_update_temp = get_rows_from_updated_property_details_api(data_update, 20, 30, ZWSID2)
    # data_update = data_update.append(data_update_temp)
    # data_update_temp = get_rows_from_updated_property_details_api(data_update, 30, 40, ZWSID3)
    # data_update = data_update.append(data_update_temp)
    # data_update_temp = get_rows_from_updated_property_details_api(data_update, 40, 50, ZWSID4)
    # data_update = data_update.append(data_update_temp)

    # Output File
    data_info.to_csv('../../input/' + house_info_filename, index=False)
    data_deep.to_csv('../../input/' + house_detail_filename, index=False)
    data_update.to_csv('../../input/' + house_updated_filename, index=False)


if __name__ == "__main__":
    main()
