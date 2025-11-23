import joblib
import os
import urllib.request

MODEL_PATH = "boston_model.pkl"
MODEL_URL = "https://github.com/andrevictorm/boston-housing-api/releases/download/model-v1/boston_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Baixando modelo do GitHub Releases...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado!")

model = joblib.load(MODEL_PATH)

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

app = FastAPI(title="Boston Housing API - André Victor")

API_KEY = "boston_2025_secret_8f9d2a1c9e7b3f6d5a4c8e2f1d0b9a8e7c6d5f4"

class HouseFeatures(BaseModel):
    CRIM: float = Field(..., example=0.00632)
    ZN: float = Field(..., example=18.0)
    INDUS: float = Field(..., example=2.31)
    CHAS: int = Field(..., example=0)
    NOX: float = Field(..., example=0.538)
    RM: float = Field(..., example=6.575)
    AGE: float = Field(..., example=65.2)
    DIS: float = Field(..., example=4.09)
    RAD: int = Field(..., example=1)
    TAX: float = Field(..., example=296.0)
    PTRATIO: float = Field(..., example=15.3)
    B: float = Field(..., example=396.9)
    LSTAT: float = Field(..., example=4.98)

@app.get("/")
def root():
    return {"message": "API do André Victor - funcionando 24/7 com modelo no GitHub Releases"}

@app.post("/predict")
def predict(houses: List[HouseFeatures], x_api_key: str = Header(alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Chave errada")
    df = pd.DataFrame([h.dict() for h in houses])
    pred = model.predict(df)
    return {"preços (mil USD)": [round(float(p), 2) for p in pred]}

from google.colab import files
files.download('main.py')
