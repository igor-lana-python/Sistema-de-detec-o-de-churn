# Predição de Churn com Machine Learning

## Sobre
API que prevê probabilidade de cancelamento de clientes usando Random Forest.

## Como rodar
```bash
pip install -r requirements.txt
uvicorn api:app --reload
## EXEMPLO
``
POST /predict
{
    "gender": "Male",
    "tenure": 2,
    "MonthlyCharges": 110
}

Resposta:
{
    "probabilidade_churn": 83,
    "vai_cancelar": "Sim"
}
``
## Tecnologias
