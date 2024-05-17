"""
Created on Friday May 17 2024

@author: Aditi Vaidya

Function that computes performance on model slices
"""
# Import libraries
import joblib
import pandas as pd
from ml.model import compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split

# Load in the data.
data = pd.read_csv("data/census_clean.csv")

# train-test split (20% test)
_, test = train_test_split(data, test_size=0.20)

# Categorical Features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Call pickled model and encoders
with open('./model/XGBoost_model1.pkl', 'rb') as f:
    model = joblib.load(f) 
with open('./model/encoder.pkl', 'rb') as f:
    encoder = joblib.load(f) 
with open('./model/lb.pkl', 'rb') as f:
    lb = joblib.load(f)

# Create slice_output file by looping all categorical features
def slice(cat_features = cat_features, slice_file_path = 'slice_output.txt'):
    with open(slice_file_path, 'w') as file:
        for cat in cat_features:
            for i in list(test[cat].unique()):
                data_slice = test[test[cat] == i]
                X, y, _, _ = process_data(data_slice,
                                        cat_features,
                                        label="salary",  
                                        encoder=encoder,
                                        lb=lb,
                                        training=False)
                y_preds = inference(model, X)
                precision, recall, fbeta = compute_model_metrics(y, y_preds)
                output = f"Feature: '{cat}', Class: '{i}' - Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}\n"        
                file.write(output)

# Run the function
if __name__ == "__main__":
    slice(cat_features = cat_features, slice_file_path = 'slice_output.txt')
