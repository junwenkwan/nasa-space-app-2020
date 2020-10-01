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

app = Flask(__name__)
weights_pth = './nn_weight/mlp_weight.pth'
model = MLP(input_size=3, output_size=1)
model.load_state_dict(torch.load(weights_pth))
model.eval()

def get_prediction(in_vector):
    in_vector = torch.from_numpy(np.array(in_vector))
    in_vector =  Variable(in_vector).float()
    outputs = model.forward(in_vector)
    predicted = (outputs >= 0.755).float()
    return predicted.cpu().numpy().tolist()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        params = data['params']
        class_id = get_prediction(in_vector=params)
        return jsonify({'class_id': class_id})


if __name__ == '__main__':
    app.run()