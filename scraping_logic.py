import requests

payload = {'api_key': '9e79d7daefa8a6ac32a651f2c5399efd','asin': 'B07BY89TQY','country': 'uk'}

response = requests.get('https://api.scraperapi.com/structured/amazon/review', params=payload)

data = response.json()

print(data)
reviews=data['reviews']