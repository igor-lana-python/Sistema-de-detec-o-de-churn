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
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 85.5,
                "TotalCharges": 1026.0,
                "is_long_term": 0,
                "high_value": 0,
                "avg_monthly_spend": 85.5,
                "senior_high_value": 0
            }
        }

# Rota de saúde (teste)
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
        # Converter para DataFrame
        dados = pd.DataFrame([cliente.dict()])
        
        # Garantir que as colunas estão na ordem correta
        colunas_esperadas = preprocessor.feature_names_in_ if hasattr(preprocessor, 'feature_names_in_') else None
        
        # Aplicar pré-processamento
        dados_processados = preprocessor.transform(dados)
        
        # Fazer predição
        proba = model.predict_proba(dados_processados)[0, 1]
        churn_pred = int(proba >= 0.5)
        
        # Determinar nível de risco
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

# Rota de informação do modelo
@app.get("/model-info")
def model_info():
    return {
        "modelo": type(model).__name__,
        "features_esperadas": preprocessor.feature_names_in_.tolist() if hasattr(preprocessor, 'feature_names_in_') else None
    }