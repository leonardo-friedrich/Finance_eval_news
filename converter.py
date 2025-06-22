import pandas as pd

# Lê o arquivo com codificação correta
df = pd.read_csv("Sentences_AllAgree.txt", sep='@', header=None, names=["sentence", "sentiment"], encoding='latin1')

# Limpeza de texto
df["sentence"] = df["sentence"].str.strip()
df["sentiment"] = df["sentiment"].str.strip().str.lower()

# Exporta para CSV
df.to_csv("FinancialPhraseBank.csv", index=False, encoding='utf-8', sep=";")
print("✅ Arquivo FinancialPhraseBank.csv criado com sucesso!")
