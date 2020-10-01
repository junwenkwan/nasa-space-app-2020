import requests
import numpy as np

arr = np.array([29.015749, 325.264507747156, 0.668088614940643])
data = {'params': arr.tolist()}

resp = requests.post("http://localhost:5000/predict",
                     json=data)

print(resp.json())

