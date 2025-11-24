from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
import urllib.request

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Iniciando Boston Housing API...")

# Carregar modelo
MODEL_PATH = "boston_model.pkl"
MODEL_URL = "https://github.com/andrevictorm/boston-housing-api/releases/download/model-v1/boston_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        print("📥 Baixando modelo...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo carregado!")
except Exception as e:
    print(f"❌ Erro no modelo: {e}")
    model = None

API_KEY = "boston_2025_secret_8f9d2a1c9e7b3f6d5a4c8e2f1d0b9a8e7c6d5f4"

class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float  
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/")
def root():
    return {"message": "Boston Housing API", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(houses: List[HouseFeatures], x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "API Key inválida")
    
    if not model:
        raise HTTPException(500, "Modelo não carregado")
    
    try:
        df = pd.DataFrame([house.dict() for house in houses])
        predictions = model.predict(df)
        
        return {
            "predictions": [
                {"house_index": i, "price": round(float(p), 2)}
                for i, p in enumerate(predictions)
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Erro: {str(e)}")

print("✅ API pronta!")
