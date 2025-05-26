# 🧠 API de Predição de Bebedores (`DRK_YN`)

Este projeto é uma API de Machine Learning desenvolvida com **FastAPI**, com o objetivo de prever se um indivíduo **consome bebidas alcoólicas** com base em variáveis biométricas e clínicas. A variável alvo utilizada é `DRK_YN` — onde:

- `0` → Não bebe
- `1` → Bebe

## 📊 Dataset

O conjunto de dados foi extraído de um repositório de saúde pública sul-coreano e contém informações sobre:

- Características físicas: `sex`, `age`, `height`, `weight`, `waistline`
- Indicadores visuais e auditivos: `sight_left`, `hear_right`, etc.
- Pressão arterial e exames de sangue: `SBP`, `DBP`, `BLDS`, `tot_chole`, `gamma_GTP`, etc.
- Variáveis de estilo de vida: `SMK_stat_type_cd`, `DRK_YN`

---

## 🧱 Estrutura do Projeto

```
API/
├── Features/
│   ├── Artefatos/
│   │   ├── ordinal.pkl
│   │   └── scaler.pkl
│   └── preprocessamento.py
├── Modelo/
│   ├── Artefatos/
│   │   └── modelo.bin
│   └── modelo.py
├── servico.py
├── treinar_modelo.py
└── teste_api_DRKYN.py
```

---

## 🧠 Técnicas e Conceitos Aplicados

### ✅ 1. **Machine Learning Supervisionado**
- Target: `DRK_YN` (bebe ou não)
- Algoritmo utilizado: `XGBoost` (via `FabricaXGB` com Abstract Factory)

### ✅ 2. **Pré-processamento**
- **Encoder Ordinal**: para transformar `sex` em valores numéricos
- **MinMaxScaler**: para normalizar as variáveis contínuas
- Os artefatos (`.pkl`) são salvos e reaproveitados na API

### ✅ 3. **Padrão de Projeto - Abstract Factory**
Utilizamos o padrão Abstract Factory para facilitar a criação de diferentes modelos ML de forma modular e desacoplada. A fábrica permite que facilmente se troque de `XGBoost` para `LightGBM`, por exemplo, sem alterar a lógica da API.

### ✅ 4. **Orientação a Objetos (POO)**
- Toda a arquitetura da API foi construída com classes
- O código foi separado por responsabilidade: `Modelo`, `Features`, `Serviço`

### ✅ 5. **Deploy com FastAPI**
- Rápida, assíncrona e leve
- Endpoint principal: `POST /predict/`

---

## 🚀 Como Rodar Localmente

### 1. Instale os pacotes necessários
```bash
pip install -r requirements.txt
```

### 2. Treine o modelo e gere os artefatos
```bash
python treinar_modelo.py
```

### 3. Inicie a API
```bash
uvicorn servico:app --reload
```

### 4. Teste a API
```bash
python teste_api_DRKYN.py
```

---

## 📈 Resultado

A API retorna:
- A predição (0 = não bebe, 1 = bebe)
- As probabilidades por classe
- Comparação com o valor real (no script de teste)

---

## 📫 Contato

Feito com 💡 por [Jheferson Warley] — focado em análise de dados e ciência de dados com aplicação prática em projetos de portfólio.
