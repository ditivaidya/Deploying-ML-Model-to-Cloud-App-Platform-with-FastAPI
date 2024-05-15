import os
import sys
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import joblib
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import dvc.api

#with dvc.api.open('/data/', 
#                  repo='https://github.com/ditivaidya/Deploying-ML-Model-to-Cloud-App-Platform-with-FastAPI.git') as f:
#    # Read data from the file-like object (f)
#    data = pd.read_csv(f)

file_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(file_dir,"./data/census_cleaned.csv"))

#data = pd.read_csv("../data/census_clean.csv")
train, test = train_test_split(data, test_size=0.20)


def test_process_data():
    '''
    test pre-process data function (process_data)
    '''
    
    cat_features = ["workclass", "sex"]
    X_train, y_train, _, _ = process_data(train,
                                          categorical_features=cat_features,
                                          label="salary",
                                          training=True)
    assert X_train.shape[1] == 6 + 2 + 7 + 6
    # Number of Continuous variables = 6
    # One-hot encoded "sex" = 2
    # One-hot encoded "workclass" = 7    
    # Number of Unprocessed categorical variables = 6
    assert X_train.shape[0] == y_train.shape[0]


def test_compute_model_metrics():
    with open('../model/XGBoost_model1.pkl', 'rb') as f:
        model = joblib.load(f) # deserialize using load()
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = joblib.load(f) # deserialize using load()
    with open('../model/lb.pkl', 'rb') as f:
        lb = joblib.load(f) # deserialize using load()        
    #model = pickle.load(open('../model/XGBoost_model1.pkl', 'rb'))
    #encoder = pickle.load(open('../model/encoder.pkl', 'rb'))
    #lb = pickle.load(open('../model/lb.pkl', 'rb'))
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",]
    X, y, _, _= process_data(test,
                             categorical_features=cat_features,
                             label="salary",
                             training=False,
                             encoder=encoder,
                             lb=lb)

    y_preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, y_preds)


    assert 0 < precision < 1
    assert 0 < recall < 1
    assert 0 < fbeta < 1

def test_inference():
    with open('../model/XGBoost_model1.pkl', 'rb') as f:
        model = joblib.load(f) # deserialize using load()
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = joblib.load(f) # deserialize using load()
    with open('../model/lb.pkl', 'rb') as f:
        lb = joblib.load(f) # deserialize using load()        
    #model = pickle.load(open('../model/XGBoost_model1.pkl', 'rb'))
    #encoder = pickle.load(open('../model/encoder.pkl', 'rb'))
    #lb = pickle.load(open('../model/lb.pkl', 'rb'))
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",]
    X, y, _, _= process_data(test,
                             categorical_features=cat_features,
                             label="salary",
                             training=False,
                             encoder=encoder,
                             lb=lb)

    y_preds = inference(model, X)

    assert y_preds.shape == y.shape
    
