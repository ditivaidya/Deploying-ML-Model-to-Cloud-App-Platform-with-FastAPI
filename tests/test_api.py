import json
from fastapi import testclient
from main import app
import pytest

client = testclient.TestClient(app)

def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Welcome to the Model Inference API"}

def test_post_0():
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
   response = client.post("/predict", data = test_input)
   assert response.status_code == 200
   assert response.json() == {'prediction': '>50K'}


def test_post_1():
   payload = {"age": 25,
        "workclass": "Private",
        "fnlgt": 287927,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Unmarried",
        "race": "Other",
        "sex": "Female",
        "capital_gain": 1000,
        "capital_loss": 88,
        "hours_per_week": 30,
        "native_country": "Columbia"
        }
   
   test_input = json.dumps(payload)
   response = client.post("/predict", data = test_input)
   assert response.status_code == 200
   assert response.json() == {'prediction': '<=50K'}

if __name__ == "__main__":
    test_get()
    test_post_0()
    test_post_1()