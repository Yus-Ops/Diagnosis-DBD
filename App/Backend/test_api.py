import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "wbc_count": 5600,
    "platelet_count": 150000,
    "neutrophils": 60,
    "lymphocytes": 30,
    "mpv": 10.5,
    "pdw": 12.3,
    "hemoglobin": 13.2,
    "hct": 40.1,
    "patient_id": "TEST123"
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)

print(response.status_code)
print(response.json())
