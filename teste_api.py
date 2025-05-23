# Criando um script Python que compara a previsão da API com o valor real (rótulo)

import requests

# ✅ Dado real (rótulo verdadeiro da amostra)
# SMK_stat_type_cd:
# 1 → Nunca fumou
# 2 → Ex-fumante
# 3 → Fumante atual

# Para fins de comparação, vamos guardar o valor real (original)
valor_real = 1.0  # Nunca fumou (classe 0)

# Criando a amostra (com o valor real incluído apenas para referência)
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
  "gamma_GTP": 35.0
}

# Envia para a API
url = "http://127.0.0.1:8000/predict/"
res = requests.post(url, json=amostra)

print("Status:", res.status_code)

try:
    resultado = res.json()
    pred = resultado["previsao"]

    # Como estamos prevendo DRK_YN (Y/N), temos um modelo binário:
    # 0 = Não bebe (DRK_YN = 'N')
    # 1 = Bebe       (DRK_YN = 'Y')
    mapeamento = {
        0: "Não bebe (DRK_YN = 'N')",
        1: "Bebe (DRK_YN = 'Y')"
    }

    print("""
            Previsão do modelo sobre a variável alvo: DRK_YN
            Target: DRK_YN (bebedor)
                - 0 → Não bebe
                - 1 → Bebe
            """)

    print(f" Previsão do modelo: Classe {pred} → {mapeamento[pred]}")
    print(f" Valor real informado: Classe {int(valor_real)} → {mapeamento[int(valor_real)]}")

    if pred == int(valor_real):
        print("✅ O modelo acertou!")
    else:
        print("❌ O modelo errou.")

    print("Probabilidades por classe:")
    for i, prob in enumerate(resultado["probabilidades"]):
        print(f"  Classe {i} → {mapeamento[i]}: {prob*100:.2f}%")

except Exception as e:
    print("❌ Erro ao interpretar resposta:")
    print(res.text)
