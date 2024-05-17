import json
from fastapi import testclient
from main import app
import pytest

client = testclient.TestClient(app)

def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Welcome to the Model Inference API"}

#def test_post_0():
#   pass


#def test_post_1():
#   pass

if __name__ == "__main__":
    test_get()