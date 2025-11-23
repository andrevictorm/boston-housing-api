import joblib
import os
import urllib.request
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

# ---------- BAIXA O MODELO AUTOMATICAMENTE ----------
MODEL_PATH = "boston_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Primeira execução no Railway: baixando modelo (~139MB)...")
    url = "https://drive.google.com/uc?export=download&id=1f8eK9j8sL8qX9v2kPqR5tY7uI9oP2mN5"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo baixado com sucesso!")

model = joblib.load(MODEL_PATH)
# ---------------------------------------------------

app = FastAPI(
    title="Boston Housing Price API - André Victor",
    description="Batch prediction + API Key + modelo carregado automaticamente",
    version="3.0"
)

API_KEY = "boston_2025_secret_8f9d2a1c9e7b3f6d5a4c8e2f1d0b9a8e7c6d5f4"

class HouseFeatures(BaseModel):
    CRIM: float = Field(..., example=0.006)
    ZN: float = Field(..., example=18.0)
    INDUS: float = Field(..., example=2.31)
    CHAS: int = Field(..., example=0, ge=0, le=1)
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
    return {"message": "API do André Victor rodando 24/7 com modelo carregado!"}

@app.post("/predict")
def predict(houses: List[HouseFeatures], x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "API Key inválida")
    
    df = pd.DataFrame([h.dict() for h in houses])
    predictions = model.predict(df)
    
    return {
        "total_predições": len(predictions),
        "preços_em_milhares_de_USD": [round(float(p), 2) for p in predictions]
    }
