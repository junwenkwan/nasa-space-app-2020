import requests
import numpy as np


data = {'top_left_latlng': [-33.2,150], 'bottom_right_latlng': [-34.2,151], 'datetime': '2020-01-01' }

resp = requests.post("http://127.0.0.1:5000/predict",
                     json=data)

print(resp.json())

