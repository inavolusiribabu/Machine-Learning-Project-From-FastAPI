from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -------------------------
# Load saved pipeline
# -------------------------
model = joblib.load("adult_income_pipeline.pkl")

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="Adult Income Prediction API")

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def home():
    return {"message": "Adult Income Prediction API is running successfully"}

# -------------------------
# Input Schema
# -------------------------
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# -------------------------
# Predict Endpoint
# -------------------------
@app.post("/predict")
def predict(data: Person):

    try:
        # Convert input to dataframe
        df = pd.DataFrame([data.model_dump()])

        # Model prediction
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        label = ">50K" if pred == 1 else "<=50K"

        return {
            "prediction": int(pred),
            "income_class": label,
            "probability_<=50K": round(float(proba[0]), 3),
            "probability_>50K": round(float(proba[1]), 3)
        }

    except Exception as e:
        return {"error": str(e)}