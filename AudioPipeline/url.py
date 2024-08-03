import requests

url = "http://127.0.0.1:9999/audio"
data = {
    "audio_path": "./test_api/"
}

response = requests.post(url, json=data)
print(response.json())
