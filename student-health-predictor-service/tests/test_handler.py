import requests
import json

with open('./sample_low.json', 'rt', encoding='utf-8') as f_in:
    data = json.load(f_in)

url = 'http://localhost:9696/predict'

result = requests.post(url, json=data).json()
print(result)