import requests
import numpy as np
import time

data = {'top_left_latlng': [-33.2,150], 'bottom_right_latlng': [-34.2,151], 'datetime': '2020-01-01' }

start = time.time()
resp = requests.post("http://127.0.0.1:5000/predict",
                     json=data)
elapsed = time.time()

print(resp.json())
print('total time taken:',elapsed-start)
