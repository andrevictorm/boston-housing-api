**Boston Housing Price Prediction API**

API com FastAPI. Predição em lote. Autenticação por chave. Modelo RandomForest. Documentação automática em Swagger e Redoc.

**Contexto**

Este projeto expõe um modelo de previsão de preços de casas de Boston como um serviço web. O modelo é um RandomForest treinado no conjunto Boston Housing. A API demonstra um fluxo completo. Treino. Serialização. Autenticação por chave. Predição em lote. Documentação automática.

**Porque este exemplo**

Base para MLOps inicial. Mostra práticas simples. Separação entre treino e serving. Contrato de entrada e saída. Validação básica. Monitorização via logs. Integração com CI e CD. Deploy em serviços como Render.

**Dados e modelo**

Conjunto Boston Housing para fins educativos. Possui limitações e risco de viés. Treino com scikit learn. Algoritmo RandomForestRegressor. Treino offline. Artefacto guardado com joblib. A API apenas carrega o modelo e faz inferência.

**Segurança**

A API exige chave no header X API Key. O valor deve vir de variável de ambiente. No repositório usa `BOSTON_API_KEY`. Em produção use um gestor de segredos. Nunca faça commit de segredos.

**Endpoints**

- POST `/predict`  
  Recebe uma lista de casas. Retorna uma lista de previsões.

**Headers**

- `X-API-Key` com a chave válida  
- `Content-Type` `application/json`

**Contrato de entrada**

Lista de objetos com chaves  
`CRIM` `ZN` `INDUS` `CHAS` `NOX` `RM` `AGE` `DIS` `RAD` `TAX` `PTRATIO` `B` `LSTAT`  
A coluna TAX é por valor total por \$10,000

**Exemplo de pedido com curl**

```bash
curl -X POST https://boston-housing-api-2.onrender.com/predict \
  -H "X-API-Key: $BOSTON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '[
    {"CRIM":0.1,"ZN":0.0,"INDUS":8.14,"CHAS":0,"NOX":0.538,"RM":6.0,"AGE":65.2,"DIS":4.0,"RAD":4,"TAX":307,"PTRATIO":21.0,"B":390.0,"LSTAT":12.0},
    {"CRIM":0.03,"ZN":25.0,"INDUS":5.13,"CHAS":0,"NOX":0.453,"RM":6.5,"AGE":45.0,"DIS":5.0,"RAD":5,"TAX":300,"PTRATIO":18.0,"B":395.0,"LSTAT":8.0}
  ]'
```

**Exemplo em Python**

```python
import requests
import os

url = "https://boston-housing-api-2.onrender.com/predict"
headers = {
    "X-API-Key": os.environ["BOSTON_API_KEY"],
    "Content-Type": "application/json",
    "User-Agent": "MyBostonApp/1.0"
}
payload = [
    {"CRIM":0.1,"ZN":0.0,"INDUS":8.14,"CHAS":0,"NOX":0.538,"RM":6.0,"AGE":65.2,"DIS":4.0,"RAD":4,"TAX":307,"PTRATIO":21.0,"B":390.0,"LSTAT":12.0}
]
r = requests.post(url, json=payload, headers=headers, timeout=20)
print(r.status_code, r.json())
```

**Resposta**

```json
{
  "predictions": [
    {"house_index": 0, "price": 24.87}
  ],
  "count": 1,
  "model": "RandomForestRegressor",
  "version": "1.0.0"
}
```

**Validação e erros**

- Campos em falta geram 422 Unprocessable Entity  
- Chave inválida gera 401 Unauthorized  
- Erros internos devolvem 500 Internal Server Error com `trace_id` nos logs

**Documentação**

- Abrir `/docs` para Swagger UI  
- Abrir `/redoc` para Redoc

**Execução local**

```bash
export BOSTON_API_KEY="boston_2025_secret_..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Teste rápido**

```bash
curl -H "X-API-Key: $BOSTON_API_KEY" http://localhost:8000/health
```

**Estrutura do projeto**

- `app/main.py` inicializa a FastAPI  
- `app/routes.py` define `/predict`  
- `app/schemas.py` define modelos Pydantic  
- `models/model.joblib` guarda o RandomForest  
- `notebooks/` contém treino e exploração  
- `tests/` contém testes de contrato

**Batch prediction**

O endpoint aceita uma lista de casas. A resposta traz um índice por casa. O cliente em Python recolhe inputs. Envia em lote. Imprime estatísticas simples.

**Observabilidade**

Logs estruturados com `logging`. Time. Nível. Rota. Latência. Integra com CloudWatch. Stackdriver. Loki.

**Limitações**

Conjunto antigo com risco de viés. O alvo está em mil dólares. O modelo não é indicado para decisão real. Uso apenas demonstrativo.

**Licença**

Uso educativo. Sem garantias.
