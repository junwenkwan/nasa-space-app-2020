import requests
import numpy as np
import time

data = {'top_left_lat_lng': [-33.2,150], 'bottom_right_lat_lng': [-34.2,151], 'date': '2020-04-04' }

start = time.time()
resp = requests.post("http://3.133.101.18:80/update_assets", json=data)
elapsed = time.time()
print('total time taken:',elapsed-start)
print(resp.json())

start = time.time()
resp = requests.post("http://3.133.101.18:80/predict", json=data)
elapsed = time.time()
print('total time taken:',elapsed-start)
print(resp.json())

start = time.time()
resp = requests.post("http://3.133.101.18:80/process_firms_data", json=data)
elapsed = time.time()
print('total time taken:',elapsed-start)
print(resp.json())

# start = time.time()
# resp = requests.post("http://127.0.0.1:5000/update_assets", json=data)
# elapsed = time.time()
# print('total time taken:',elapsed-start)
# print(resp.json())

# start = time.time()
# resp = requests.post("http://127.0.0.1:5000/predict", json=data)
# elapsed = time.time()
# print('total time taken:',elapsed-start)
# print(resp.json())