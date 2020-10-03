import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, send_file, send_from_directory
from model.mlp import MLP
import torch
from torch.autograd import Variable
import numpy as np
from utils.utils import get_rainfall, get_solar_insolation, get_temperature
import os
import requests
from skimage import io
import shutil

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
    return predicted.cpu().numpy()

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

        threshold = 0.2
        
        lat_arr = np.arange(top_left_lat, bottom_right_lat, -threshold)
        lng_arr = np.arange(top_left_lng, bottom_right_lng, threshold)

        coords_arr = []

        for x in lat_arr:
            for y in lng_arr:
                coords_arr.append([round(x,1), round(y,1)])

        temperature = get_temperature(coords_arr,datetime)
        solar_insolation = get_solar_insolation(coords_arr, datetime)
        rainfall = get_rainfall(coords_arr, datetime)

        results = []
        for i, j, k in list(zip(temperature, solar_insolation,rainfall)):
            params = np.array([i,j,k])
            class_id = get_prediction(in_vector=params)
            results.append(class_id)

        return jsonify({'class_id': results})

@app.route('/update_assets', methods=['POST'])
def update_assets():
    if request.method == 'POST':
        data = request.json

        datetime = data['datetime']

        # Delete existing files
        for filename in os.listdir('./assets'):
            file_path = os.path.join('./assets', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        app.logger.info('Deleted existing files')    
        
        # Download temperature data
        temperature_url = 'https://neo.sci.gsfc.nasa.gov/archive/csv/MOD_LSTD_D/MOD_LSTD_D_{}.CSV.gz'.format(datetime)
        temperature_filename = os.path.join('./assets',temperature_url.split("/")[-1].split('.')[-3]+'.CSV')
        with open(temperature_filename, "wb") as f:
            r = requests.get(temperature_url)
            f.write(r.content)

        app.logger.info('Land temperature data successfully downloaded')

        # Download solar_insolation data
        solar_url = 'https://neo.sci.gsfc.nasa.gov/archive/rgb/CERES_INSOL_D/CERES_INSOL_D_{}.PNG'.format(datetime)
        img = io.imread(solar_url)  
        io.imsave(os.path.join('./assets',solar_url.split('/')[-1]), img) 

        app.logger.info('Solar insolation data successfully downloaded')

        # Download rainfall data
        year, month, day = datetime.split('-')
        parent_dir = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/{}/{}/'.format(year,month)

        filename = '3B-DAY.MS.MRG.3IMERG.{}-S000000-E235959.V06.nc4'.format(datetime.replace('-', ''))

        rainfall_url = parent_dir + filename
        cmd = 'wget --auth-no-challenge=on --user=kckhoo --password=\'NaSa929347\' --content-disposition --directory-prefix=\'./assets\' ' + rainfall_url

        os.system(cmd)

        app.logger.info('Rainfall data successfully downloaded')
        
        # Download FIRMS
        url = 'https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/FIRMS/noaa-20-viirs-c2/Australia_NewZealand/J1_VIIRS_C2_Australia_NewZealand_VJ114IMGTDL_NRT_2019338.txt'
        cmd = 'wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=4 --directory-prefix=\'./assets\' --header \'Authorization: Bearer 7BAC12AC-0536-11EB-AB4A-A082BEDF9A3A\' ' + url
        os.system(cmd)

        app.logger.info('FIRMS data successfully downloaded')
        status = 1

        return jsonify({'update_assets': status})

@app.route('/get_firms_data', methods=['POST'])
def get_firms_data():
    filename = 'J1_VIIRS_C2_Australia_NewZealand_VJ114IMGTDL_NRT_2019338.txt'
    dir = os.path.join('./assets','FIRMS','noaa-20-viirs-c2','Australia_NewZealand')

    try:
        return send_from_directory(dir, filename=filename)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
