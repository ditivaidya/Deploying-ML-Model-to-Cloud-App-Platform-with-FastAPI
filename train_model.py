# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import logging
import joblib
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _= process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
save_model_path = 'model/XGBoost_model1.pkl'
save_encoder_path = 'model/encoder.pkl'
save_lb_path = 'model/lb.pkl'

logging.info('Training model')
model, best_params = train_model(X_train, y_train)

print(f"Best Parameters of model: {best_params}")

logging.info('Saving model')
joblib.dump(model, save_model_path)
joblib.dump(encoder, save_encoder_path)
joblib.dump(lb, save_lb_path)

y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print(f"Model Performance: Precision Score {precision}, Recall Score {recall}, fbeta {fbeta}")
