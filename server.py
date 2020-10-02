import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from model.mlp import MLP
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
import requests

app = Flask(__name__)
weights_pth = './nn_weight/mlp_weight.pth'
model = MLP(input_size=3, output_size=1)
model.load_state_dict(torch.load(weights_pth, map_location=torch.device('cpu')))
model.eval()

def get_prediction(in_vector):
    in_vector = torch.from_numpy(np.array(in_vector))
    in_vector =  Variable(in_vector).float()
    outputs = model.forward(in_vector)
    predicted = (outputs >= 0.755).float()
    return predicted.cpu().numpy().tolist()

def get_temperature(lat_arr, lng_arr, datetime):
    temperature = []

    # Define column and index labels for renaming later
    column_labels = np.around(np.arange(0.1,180.1,0.1), 1)
    column_labels = np.concatenate((-np.flip(column_labels), column_labels))
    index_labels = np.around(np.arange(0.1,90.1,0.1), 1)
    index_labels = np.concatenate((np.flip(index_labels), -index_labels))

    # Define url for csv file containing land surface temperature
    nasa_url = 'https://neo.sci.gsfc.nasa.gov/archive/csv/MOD_LSTD_D/MOD_LSTD_D_{}.CSV.gz'.format(datetime)

    # Retrive csv file using url
    filename = nasa_url.split("/")[-1].split(".")[-3]

    with open(filename, "wb") as f:
        r = requests.get(nasa_url)
        f.write(r.content)

    df_temperature = pd.read_csv(filename, compression='gzip', header=None)

    # Rename index and column of the dataframe for the ease of accessing cell value
    df_temperature.columns = column_labels
    df_temperature.index = index_labels

    for lat, lng in list(zip(lat_arr, lng_arr)):
        temperature.append(df_temperature.loc[round(lat, 1), round(lng, 1)])

    os.remove(filename)
    return temperature

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json

        # top_left_latlng and bottom_right_latlng is expected to be a list in the format of ([lat, lng])
        top_left_latlng = data['top_left_latlng'] 
        bottom_right_latlng = data['bottom_right_latlng'] 

        # date is expected to be a string
        datetime = data['datetime']

        top_left_lat = top_left_latlng[0]
        top_left_lng = top_left_latlng[1]
        bottom_right_lat = bottom_right_latlng[0]
        bottom_right_lng = bottom_right_latlng[1]

        threshold = 0.1
        
        lat_arr = np.arange(top_left_lat, bottom_right_lat, -threshold)
        lng_arr = np.arange(top_left_lng, bottom_right_lng, threshold)

        temperature = get_temperature(lat_arr,lng_arr,datetime)

        params = np.array([29.015749, 325.264507747156, 0.668088614940643])
        class_id = get_prediction(in_vector=params)
        return jsonify({'class_id': class_id})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
