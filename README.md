# nasa-space-app-back-end
Front-end: [https://github.com/junwenkwan/nasa-space-app-front-end](https://github.com/junwenkwan/nasa-space-app-front-end)

## Flask Development
```bash
FLASK_ENV=development FLASK_APP=server.py flask run
```

## AWS EC2
```bash
ssh -i "the-universe-academy.pem" ubuntu@ec2-3-133-101-18.us-east-2.compute.amazonaws.com
```

## Virtual Environment
```bash
python3 -m venv venv && source venv/bin/activate
```

## RPC APIs
```bash
POST http://127.0.0.1:5000/update_assets
{'datetime': 'YYYY-MM-DD' }

Response:
{'update_assets': 1}
```

```bash
POST http://127.0.0.1:5000/predict
{'top_left_lat_lng': [x1,y1], 'bottom_right_lat_lng': [x2,y2], 'datetime': 'YYYY-MM-DD' }

Response:
{'class_id': list, 'latitude': list(latitude), 'longitude': list(longitude)}
```

```bash
POST http://127.0.0.1:5000/process_firms_data
{'date': 'YYYY-MM-DD', 'index': int, 'country': 'Australia_NewZealand'}
 
Response:
{'latitude': latitude, 'longitude': longitude, 'bright_ti4': bright_ti4, \
 'track': track, 'date': date, 'time':time, 'confidence': confidence }
```

## Train Neural Network
```bash
python3 training/training_script.py --weights-path ./nn_weight/new.pth \
                                    --csv-path ./dataset/master_dataset.csv 
```
