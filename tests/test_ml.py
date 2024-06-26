"""
Created on Friday May 17 2024

@author: Aditi Vaidya

Unit tests that ensure all model functions are working as expected 
"""
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Load in the data.
data = pd.read_csv("./data/census_clean.csv")

# Train/Test data split
train, test = train_test_split(data, test_size=0.20)

# Call pickled model and encoders
with open('./model/XGBoost_model1.pkl', 'rb') as f:
    model = joblib.load(f) # deserialize using load()
with open('./model/encoder.pkl', 'rb') as f:
    encoder = joblib.load(f) # deserialize using load()
with open('./model/lb.pkl', 'rb') as f:
    lb = joblib.load(f) # deserialize using load()        

# Categorical Features
cat_features = ["workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",]

# Categorical Features for testing
cat_features_test = ["workclass", "sex"]

def test_process_data():
    '''
    test pre-process data function (process_data)
    '''
    X_train, y_train, _, _ = process_data(train,
                                          categorical_features=cat_features_test,
                                          label="salary",
                                          training=True)
    # Number of Continuous variables = 6
    # One-hot encoded "sex" = 2
    # One-hot encoded "workclass" = 7    
    # Number of Unprocessed categorical variables = 6
    assert X_train.shape[1] == 6 + 2 + 7 + 6
    assert X_train.shape[0] == y_train.shape[0]


def test_compute_model_metrics():
    '''
    Function to test compute_model_metrics() gives all the right metrics in the range of 0 and 1
    '''
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
    '''
    Function to test inference()
    '''
    X, y, _, _= process_data(test,
                             categorical_features=cat_features,
                             label="salary",
                             training=False,
                             encoder=encoder,
                             lb=lb)
    y_preds = inference(model, X)
    assert y_preds.shape == y.shape