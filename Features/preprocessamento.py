from pathlib import Path
import joblib
import pandas as pd
import numpy as np

class Preprocessador:
    def __init__(self):
        
        self.base_dir = Path(__file__).resolve().parent
        self.artefatos_dir = self.base_dir / "Artefatos"
        
        # Caminhos dos arquivos
        self.encoder_path = self.artefatos_dir / "ordinal.pkl"
        self.scaler_path = self.artefatos_dir / "scaler.pkl"
        
       
        self.encoder = joblib.load(self.encoder_path)
        self.scaler = joblib.load(self.scaler_path)

    def transformar(self, dados: dict) -> np.ndarray:
        df = pd.DataFrame([dados])

            # Codifica colunas categóricas
        df[['sex', 'DRK_YN']] = self.encoder.transform(df[['sex', 'DRK_YN']])

        # Remove colunas que não foram usadas no treino
        if 'SMK_stat_type_cd' in df.columns:
            df = df.drop(columns=['SMK_stat_type_cd'])

        
        dados_escalados = self.scaler.transform(df)

        return dados_escalados
