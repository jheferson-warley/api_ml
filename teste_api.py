import requests
import json

url = "http://127.0.0.1:8000/predict/"

amostra = {
  "sex": "Male",
  "age": 45,
  "height": 175,
  "weight": 70,
  "waistline": 85.0,
  "sight_left": 1.0,
  "sight_right": 1.0,
  "hear_left": 1.0,
  "hear_right": 1.0,
  "SBP": 120.0,
  "DBP": 80.0,
  "BLDS": 90.0,
  "tot_chole": 180.0,
  "HDL_chole": 55.0,
  "LDL_chole": 100.0,
  "triglyceride": 150.0,
  "hemoglobin": 14.0,
  "urine_protein": 1.0,
  "serum_creatinine": 1.1,
  "SGOT_AST": 30.0,
  "SGOT_ALT": 25.0,
  "gamma_GTP": 35.0,
  "SMK_stat_type_cd": 1.0,
  "DRK_YN": "Y"
}

response = requests.post(url, json=amostra)

print("Status:", response.status_code)

try:
    resultado = response.json()
    print(f"\n🧠 Previsão: {resultado['previsao']}")
    print("📊 Probabilidades por classe:")
    for i, prob in enumerate(resultado["probabilidades"]):
        print(f"  Classe {i}: {prob*100:.2f}%")
except Exception as e:
    print("❌ Erro ao interpretar resposta:")
    print(response.text)
