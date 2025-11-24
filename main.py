from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
import urllib.request

app = FastAPI()

print("🚀 Iniciando API...")

# Carregar modelo
MODEL_PATH = "boston_model.pkl"
MODEL_URL = "https://github.com/andrevictorm/boston-housing-api/releases/download/model-v1/boston_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo carregado!")
except Exception as e:
    print(f"❌ Erro: {e}")
    model = None

class HouseFeatures(BaseModel):
    CRIM: float; ZN: float; INDUS: float; CHAS: int; NOX: float
    RM: float; AGE: float; DIS: float; RAD: int; TAX: float
    PTRATIO: float; B: float; LSTAT: float

@app.get("/")
def home():
    return {"message": "Boston Housing API"}

@app.post("/predict")
async def predict(request: Request):
    # Verificar API key manualmente
    api_key = request.headers.get("x-api-key")
    if api_key != "boston_2025_secret_8f9d2a1c9e7b3f6d5a4c8e2f1d0b9a8e7c6d5f4":
        raise HTTPException(401, "API Key inválida")
    
    if not model:
        raise HTTPException(500, "Modelo não carregado")
    
    try:
        data = await request.json()
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        
        return {"predictions": [round(float(p), 2) for p in predictions]}
    except Exception as e:
        raise HTTPException(400, f"Erro: {str(e)}")

print("✅ API rodando!")
