import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'T':2,'TM':2,'Tm':2,'SLP':2,'H':2,'V':2,'VV':2,'VM':2 })

print(r.json())