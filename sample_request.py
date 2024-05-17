import requests
import json

payload = {"age": 30,
        "workclass": "Local-gov",
        "fnlgt": 287927,
        "education": "Masters",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "Other",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "Columbia"
        }

test_input = json.dumps(payload)

response = requests.post("https://deploying-ml-model-to-cloud-app-platform.onrender.com/predict", test_input)

print(response.status_code)
print(response.json())