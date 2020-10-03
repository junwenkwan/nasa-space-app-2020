import json
import numpy as np
import pandas as pd
import os
import requests
import cv2 
from skimage import io
import netCDF4 as nc

def get_temperature(coords_arr, datetime):
    temperature = []

    # Define column and index labels for renaming later
    column_labels = np.around(np.arange(0.1,180.1,0.1), 1)
    column_labels = np.concatenate((-np.flip(column_labels), column_labels))
    index_labels = np.around(np.arange(0.1,90.1,0.1), 1)
    index_labels = np.concatenate((np.flip(index_labels), -index_labels))

    filename = 'MOD_LSTD_D_{}.CSV'.format(datetime)
    filename = os.path.join('./assets',filename)

    df_temperature = pd.read_csv(filename, compression='gzip', header=None)

    # Rename index and column of the dataframe for the ease of accessing cell value
    df_temperature.columns = column_labels
    df_temperature.index = index_labels

    for lat, lng in coords_arr:
        temperature.append(df_temperature.loc[round(lat, 1), round(lng, 1)])

    return temperature

def get_solar_insolation(coords_arr, datetime):
    solar_insolation = []

    # Define column and index labels for renaming later
    column_labels = np.around(np.arange(0.25,180.25,0.25), 2)
    column_labels = np.concatenate((-np.flip(column_labels), column_labels))
    index_labels = np.around(np.arange(0.25,90.25,0.25), 2)
    index_labels = np.concatenate((np.flip(index_labels), -index_labels))

    filename = 'CERES_INSOL_D_{}.PNG'.format(datetime)
    filename = os.path.join('./assets',filename)

    # Retrive PNG using url
    img = io.imread(filename)

    # Remove alpha channel of PNG image
    if len(img.shape) > 2 and img.shape[2] == 4:
        # Convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Compress 3D into 2D by computing the euclidean distance for each pixel
    img_norm = np.linalg.norm(img, axis=2)

    # Convert numpy array to data frame
    df_solar_insolation = pd.DataFrame(data=img_norm)

    # Rename index and column of the dataframe for the ease of accessing cell value
    df_solar_insolation.columns = column_labels
    df_solar_insolation.index = index_labels

    for lat, long in coords_arr:
        # Round latitude and longitude to nearest 0.25
        lat_new = round(lat*4)/4
        long_new = round(long*4)/4

        solar_insolation.append(df_solar_insolation.loc[lat_new, long_new])
    
    return solar_insolation

def get_rainfall(coords_arr, datetime):
    rainfall = []

    # Define column and index labels for renaming later
    column_labels = np.around(np.arange(0.1,180.1,0.1), 1)
    column_labels = np.concatenate((-np.flip(column_labels), column_labels))
    index_labels = np.around(np.arange(0.1,90.1,0.1), 1)
    index_labels = np.concatenate((np.flip(index_labels), -index_labels))

    # Define name of the file containing rainfall
    filename = '3B-DAY.MS.MRG.3IMERG.{}-S000000-E235959.V06.nc4'.format(datetime.replace('-', ''))
    filename = os.path.join('./assets',filename)
  
    ds = nc.Dataset(filename)
    
    # Retrive rainfall array
    arr_rainfall = np.rot90(ds['precipitationCal'][:].squeeze())

    # Replace missing value in rainfall array with NaN
    unmasked_arr_rainfall = arr_rainfall.data
    unmasked_arr_rainfall[unmasked_arr_rainfall == -9999.9] = np.nan

    # Convert numpy array to data frame
    df_rainfall = pd.DataFrame(data=arr_rainfall)

    # Rename index and column of the dataframe for the ease of accessing cell value
    df_rainfall.columns = column_labels
    df_rainfall.index = index_labels

    for lat, long in coords_arr:
        rainfall.append(df_rainfall.loc[round(lat, 1), round(long, 1)])

    return rainfall