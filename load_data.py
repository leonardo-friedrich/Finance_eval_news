import pandas as pd

def load_financial_phrasebank(filepath, sample_size=100):
    df = pd.read_csv(filepath, sep=';')
    df = df[['sentence', 'sentiment']]
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df
