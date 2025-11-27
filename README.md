Contexto

Este projeto expõe um modelo de previsão de preços de casas de Boston como um serviço web. Usei FastAPI para criar um endpoint de previsão. O modelo é um RandomForest treinado com o conjunto Boston Housing. O objetivo é demonstrar um fluxo completo. Treino. serialização. API com autenticação por chave. predição em lote. documentação automática.

Porque este exemplo

Serve como base para projetos de MLOps. Mostra boas práticas simples. Separação entre treino e serving. contrato de entrada e saída. validação básica. monitorização possível via logs. Pode ser ligado a CI. CD. Pode ser publicado em serviços como Render.

Dados e modelo

O conjunto Boston Housing é usado para fins educativos. Foi muito utilizado em tutoria. Tem limitações e riscos de viés. Usei scikit learn. RandomForestRegressor. Treino offline. Guardei o artefacto com joblib. A API apenas carrega o modelo e faz inferência.

Segurança

A API pede uma chave no header X API Key. O valor deve vir de variável de ambiente. No repositório uso `BOSTON_API_KEY`. Em produção guarde a chave no gestor de segredos. Nunca faça commit de segredos.

Endpoints

POST `/predict`. Recebe uma lista de casas. Retorna uma lista de previsões.

Headers

`X-API-Key` com a chave válida.  
`Content-Type` `application/json`.

Contrato de entrada

Lista de objetos com estas chaves. `CRIM` `ZN` `INDUS` `CHAS` `NOX` `RM` `AGE` `DIS` `RAD` `TAX` `PTRATIO` `B` `LSTAT`.

Exemplo de pedido

```bash
curl -X POST https://boston-housing-api-2.onrender.com/predict \
  -H "X-API-Key: $BOSTON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '[
    {"CRIM":0.1,"ZN":0.0,"INDUS":8.14,"CHAS":0,"NOX":0.538,"RM":6.0,"AGE":65.2,"DIS":4.0,"RAD":4,"TAX":307,"PTRATIO":21.0,"B":390.0,"LSTAT":12.0},
    {"CRIM":0.03,"ZN":25.0,"INDUS":5.13,"CHAS":0,"NOX":0.453,"RM":6.5,"AGE":45.0,"DIS":5.0,"RAD":5,"TAX":300,"PTRATIO":18.0,"B":395.0,"LSTAT":8.0}
  ]'
```

Exemplo em Python

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

Resposta

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

Validação e erros

Campos em falta geram `422 Unprocessable Entity`. Chave inválida gera `401 Unauthorized`. Erros internos devolvem `500 Internal Server Error` com um `trace_id` nos logs.

Documentação

Abrir `/docs` para Swagger UI. Abrir `/redoc` para Redoc.

Execução local

```bash
export BOSTON_API_KEY="boston_2025_secret_..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Testes rápidos

```bash
curl -H "X-API-Key: $BOSTON_API_KEY" http://localhost:8000/health
```

Estrutura

`app/main.py` inicia a FastAPI.  
`app/routes.py` define `/predict`.  
`app/schemas.py` define o modelo Pydantic.  
`models/model.joblib` guarda o RandomForest.  
`notebooks/` contém treino e exploração.  
`tests/` contém testes de contrato.

Batch prediction

O endpoint aceita uma lista de casas. A API devolve um índice por casa. A aplicação de exemplo em Python recolhe inputs. envia em lote. imprime estatísticas simples.

Observabilidade

Logs estruturados com `logging`. Campos de tempo. nível. rota. tempo de resposta. Pode ligar a CloudWatch. Stackdriver. ou Loki.

Limitações

Conjunto de dados antigo. risco de viés. O alvo está em mil dólares. O modelo não é aconselhado para decisão real. Use apenas como demonstração técnica.

Licença

Uso educativo. Sem garantias.
