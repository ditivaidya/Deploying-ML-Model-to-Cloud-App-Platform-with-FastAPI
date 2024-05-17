# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from ml.model import inference  
from ml.data import process_data
import os
import joblib 

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Create FastAPI instance
app = FastAPI()

with open('./model/XGBoost_model1.pkl', 'rb') as f:
    model = joblib.load(f) 
with open('./model/encoder.pkl', 'rb') as f:
    encoder = joblib.load(f) 
with open('./model/lb.pkl', 'rb') as f:
    lb = joblib.load(f)

# Pydantic model for input data
class InputData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example='Self-emp-inc')
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Assoc-acdm")
    education_num: int = Field(..., example=12) 
    marital_status: str = Field(..., example="Divorced")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=1340)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")


# Welcome message at the root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Model Inference API"}

# POST endpoint for model inference
@app.post("/predict")
async def predict(data: InputData):
    all_features_hyph = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    all_features_und = [col.replace("-", "_") for col in all_features_hyph]
    input_conv = dict(zip(all_features_und, all_features_hyph))
    
    data_dictionary = (({input_conv[k]: v for k, v in (json.loads(data).items()) if k in input_conv}))
    input_df = pd.DataFrame(data_dictionary, index=[0])

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

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X)
    model_result = lb.inverse_transform(y_pred)[0]
    return {"prediction": model_result}

# Run the application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)