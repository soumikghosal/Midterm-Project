import requests

url = "http://localhost:9696/predict"

client = {"gender": "Male",
          "age": 33.0,
          "hypertension": 0,
          "heart_disease": 0,
          "smoking_history": "former",
          "bmi": 35.81,
          "hba1c_level": 5.8,
          "blood_glucose_level": 140}
response = requests.post(url, json=client).json()

print(response)