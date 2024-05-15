from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define the XGBoost classifier
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, np.sum(y_train == 0) / np.sum(y_train == 1)]  # Adjusting for class imbalance
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=5, verbose=1, n_jobs=-1)

    # Fit GridSearchCV to find the best parameters
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # Train the best estimator on the full training set
    best_estimator.fit(X_train, y_train)

    return best_estimator, best_params



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
    
