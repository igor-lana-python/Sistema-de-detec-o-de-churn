from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any

# Criar app
app = FastAPI(
    title="API de Predição de Churn",
    description="Prevê a probabilidade de um cliente cancelar o serviço",
    version="1.0.0"
)

# Carregar modelo e pré-processador
try:
    model = joblib.load('models/churn_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("✅ Modelo e pré-processador carregados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar: {e}")

# Definir estrutura dos dados de entrada
class Cliente(BaseModel):
    gender: str
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    is_long_term: int
    high_value: int
    avg_monthly_spend: float
    senior_high_value: int

# Rota de saúde
@app.get("/")
def home():
    return {
        "mensagem": "API de Predição de Churn - Funcionando!",
        "instrucoes": "Use POST /predict com os dados do cliente"
    }

# Rota de predição
@app.post("/predict")
def predict(cliente: Cliente) -> Dict[str, Any]:
    try:
        dados = pd.DataFrame([cliente.dict()])
        dados_processados = preprocessor.transform(dados)
        proba = model.predict_proba(dados_processados)[0, 1]
        churn_pred = int(proba >= 0.5)

        if proba < 0.3:
            risco = "Baixo"
        elif proba < 0.7:
            risco = "Médio"
        else:
            risco = "Alto"

        return {
            "probabilidade_churn": round(proba * 100, 2),
            "vai_cancelar": "Sim" if churn_pred else "Não",
            "nivel_risco": risco,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")
