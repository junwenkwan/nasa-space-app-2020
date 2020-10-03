# nasa-space-app-back-end
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
```

```bash
POST http://127.0.0.1:5000/predict
{'top_left_latlng': [x1,y1], 'bottom_right_latlng': [x2,y2], 'datetime': 'YYYY-MM-DD' }
```

```bash
POST http://127.0.0.1:5000/process_firms_data
```
