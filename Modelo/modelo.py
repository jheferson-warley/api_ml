from abc import ABC, abstractmethod
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE_DIR = Path(__file__).resolve().parent
ARTEFATOS_DIR = BASE_DIR / "Artefatos"

class ModeloClassificador(ABC):
    @abstractmethod
    def treinar(self, X, y): pass

    @abstractmethod
    def prever(self, X): pass

    @abstractmethod
    def salvar(self, caminho: Path): pass

    @abstractmethod
    def carregar(self, caminho: Path): pass


class ModeloXGB(ModeloClassificador):
    def __init__(self):
        self.model = XGBClassifier(eval_metric="mlogloss")

    def treinar(self, X, y):
        self.model.fit(X, y)

    def prever(self, X):
        return self.model.predict(X)

    def salvar(self, caminho: Path = ARTEFATOS_DIR / "modelo.bin"):
        joblib.dump(self.model, caminho)

    def carregar(self, caminho: Path = ARTEFATOS_DIR / "modelo.bin"):
        self.model = joblib.load(caminho)


class ModeloLGBM(ModeloClassificador):
    def __init__(self):
        self.model = LGBMClassifier()

    def treinar(self, X, y):
        self.model.fit(X, y)

    def prever(self, X):
        return self.model.predict(X)

    def salvar(self, caminho: Path = ARTEFATOS_DIR / "modelo.bin"):
        joblib.dump(self.model, caminho)

    def carregar(self, caminho: Path = ARTEFATOS_DIR / "modelo.bin"):
        self.model = joblib.load(caminho)


class FabricaModelos(ABC):
    @abstractmethod
    def criar_modelo(self) -> ModeloClassificador: pass


class FabricaXGB(FabricaModelos):
    def criar_modelo(self):
        return ModeloXGB()


class FabricaLGBM(FabricaModelos):
    def criar_modelo(self):
        return ModeloLGBM()
