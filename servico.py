from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from Features.preprocessamento import Preprocessador
from Modelo.modelo import ModeloXGB
from pathlib import Path
import numpy as np

app = FastAPI()

base_dir = Path(__file__).resolve().parent
modelo_path = base_dir / "Modelo" / "Artefatos" / "modelo.bin"

class DadosEntrada(BaseModel):
    sex: str
    age: int
    height: int
    weight: int
    waistline: float
    sight_left: float
    sight_right: float
    hear_left: float
    hear_right: float
    SBP: float
    DBP: float
    BLDS: float
    tot_chole: float
    HDL_chole: float
    LDL_chole: float
    triglyceride: float
    hemoglobin: float
    urine_protein: float
    serum_creatinine: float
    SGOT_AST: float
    SGOT_ALT: float
    gamma_GTP: float
    SMK_stat_type_cd: float
    DRK_YN: str

class ServicoPreditor:
    def __init__(self):
        self.preprocessador = Preprocessador()
        self.modelo = ModeloXGB()

        # Caminho robusto do modelo
        modelo_path = Path(__file__).resolve().parent / "Modelo" / "Artefatos" / "modelo.bin"
        self.modelo.carregar(modelo_path)

        
        self.previsoes: Dict[int, Dict] = {}

    def prever(self, dados: Preprocessador) -> Dict:
        id_previsao = max(self.previsoes.keys(), default=0) + 1

         
        dados_modificados = dados

        # Pré-processa com os valores originais
        dados_np = self.preprocessador.transformar(dados_modificados.dict())

       
        probs = self.modelo.model.predict_proba(dados_np)[0]
        pred = int(np.argmax(probs)) 

        # Salva no dicionário
        self.previsoes[id_previsao] = {
            "input": dados_modificados.dict(),
            "previsao": pred
        }

        return {
            "id_previsao": id_previsao,
            "previsao": pred,
            "probabilidades": probs.tolist()
        }
servico = ServicoPreditor()

@app.post("/predict/")
async def prever_endpoint(dados: DadosEntrada):
    return servico.prever(dados)

@app.get("/")
def raiz():
    return {"mensagem": "Ambiente funcionando!"}