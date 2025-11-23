from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd

# Carrega o modelo
model = joblib.load('/content/model/boston_model.pkl')

app = FastAPI(
    title="API Boston Housing - Produção",
    description="Batch prediction + API Key (produção)",
    version="3.0.0"
)

# MUDE ESSA CHAVE PARA UMA FORTE! (depois te mostro como gerar uma aleatória)
API_KEY = "boston_2025_secret_8f9d2a1c9e7b3f6d5a4c8e2f1d0b9a8e7c6d5f4"  

class HouseFeatures(BaseModel):
    CRIM: float = Field(..., example=0.00632)
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
async def root():
    return {"message": "API protegida com API Key + batch. Acesse /docs"}

@app.post("/predict")
async def predict_price(
    houses: List[HouseFeatures],
    x_api_key: str = Header(..., alias="X-API-Key")
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    if len(houses) == 0:
        raise HTTPException(status_code=400, detail="Lista vazia")
    
    df = pd.DataFrame([h.dict() for h in houses])
    predictions = model.predict(df)
    
    return {
        "total_predictions": len(predictions),
        "prices_thousands_usd": [round(float(p), 2) for p in predictions],
        "model": "RandomForestRegressor",
        "status": "success"
    }
