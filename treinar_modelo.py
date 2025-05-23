from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd
import joblib
from pathlib import Path
from Modelo.modelo import FabricaXGB


df = pd.read_parquet("smoking_drinking.parquet")
df[df.isnull().any(axis=1)]
df.drop_duplicates(inplace=True)
"""
y = df['SMK_stat_type_cd'] - 1  # transforma [1,2,3] em [0,1,2]
X = df.drop(columns=['SMK_stat_type_cd'])

# Pré-processamento
encoder = OrdinalEncoder()
X[["sex", "DRK_YN"]] = encoder.fit_transform(X[["sex", "DRK_YN"]])

"""

# Definir target e features
y = df['DRK_YN']  # Novo target
X = df.drop(columns=['DRK_YN', 'SMK_stat_type_cd'])  # Remove DRK_YN das features



# ===============================
# 3. Codificação
# ===============================
# Codifica apenas 'sex' (categorical)
encoder = OrdinalEncoder()
X[["sex"]] = encoder.fit_transform(X[["sex"]])

# Codifica o target DRK_YN (Y/N → 1/0)
y = OrdinalEncoder().fit_transform(y.values.reshape(-1, 1)).ravel()

# ===============================
# 4. Escalonamento
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. Treinamento do modelo
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

modelo = FabricaXGB().criar_modelo()
modelo.treinar(X_train, y_train)

# ===============================
# 6. Salvamento dos artefatos
# ===============================
artefatos_modelo = Path("Modelo/Artefatos")
artefatos_modelo.mkdir(parents=True, exist_ok=True)
modelo.salvar(artefatos_modelo / "modelo.bin")

artefatos_features = Path("Features/Artefatos")
artefatos_features.mkdir(parents=True, exist_ok=True)
joblib.dump(encoder, artefatos_features / "ordinal.pkl")
joblib.dump(scaler, artefatos_features / "scaler.pkl")



