import joblib
import os
import urllib.request
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import uvicorn

# CONFIGURAÇÃO ROBUSTA para Railway
app = FastAPI(
    title="Boston Housing API - André Victor",
    description="API de previsão de preços de casas em Boston - Machine Learning",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS para evitar problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Inicializando API Boston Housing...")

# Carregar modelo
MODEL_PATH = "boston_model.pkl"
MODEL_URL = "https://github.com/andrevictorm/boston-housing-api/releases/download/model-v1/boston_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        print("📥 Baixando modelo do GitHub Releases...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modelo baixado!")
    
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None

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
async def root():
    return {
        "message": "🚀 Boston Housing API - André Victor",
        "status": "online", 
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "predict": "POST /predict (com header X-API-Key)"
        },
        "version": "3.0.0"
    }

@app.get("/health")
async def health_check():
    model_status = "loaded" if model else "error"
    return {
        "status": "healthy", 
        "model": model_status,
        "service": "Boston Housing API"
    }

# 🔥 CORREÇÃO: Mudar para @app.api_route para aceitar ambos POST e OPTIONS (CORS)
@app.api_route("/predict", methods=["POST", "OPTIONS"])
async def predict(
    houses: List[HouseFeatures], 
    x_api_key: str = Header(..., alias="X-API-Key")
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    
    if not houses:
        raise HTTPException(status_code=400, detail="Lista de casas vazia")
    
    try:
        # Converter para DataFrame
        df = pd.DataFrame([house.dict() for house in houses])
        
        # Fazer previsão
        predictions = model.predict(df)
        
        return {
            "total_predictions": len(predictions),
            "predictions": [
                {
                    "house_index": i,
                    "predicted_price_usd_thousands": round(float(price), 2)
                }
                for i, price in enumerate(predictions)
            ],
            "currency": "USD",
            "unit": "thousands"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")

print("✅ API Boston Housing inicializada com sucesso!")

# Para execução local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
