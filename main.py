import pandas as pd
from load_data import load_financial_phrasebank
from prompts import generate_zero_shot_prompt, generate_few_shot_prompt
from inference import query_openai, get_fingpt_pipeline, query_fingpt
from sklearn.metrics import classification_report
import openai

def run_experiment_zero_shot(df):
    preds = []
    for _, row in df.iterrows():
        prompt = generate_zero_shot_prompt(row['sentence'])
        label = query_openai(prompt)
        preds.append(label)
    df['pred'] = preds
    return df

def run_experiment_few_shot(df, n_examples=3):
    examples = df.sample(n=n_examples).to_dict(orient='records')
    example_indices = df.sample(n=n_examples).index
    test_df = df.drop(example_indices).reset_index(drop=True)
    preds = []
    for _, row in test_df.iterrows():
        prompt = generate_few_shot_prompt(row['sentence'], examples)
        label = query_openai(prompt)
        preds.append(label)
    test_df['pred'] = preds
    return test_df

def evaluate(df):
    print(classification_report(df['sentiment'], df['pred'], zero_division=0))

if __name__ == "__main__":
    df = load_financial_phrasebank("FinancialPhraseBank.csv", sample_size=50)
    print("🔹 Zero-shot OpenAI")
    result = run_experiment_zero_shot(df)
    evaluate(result)

    print("🔹 FinGPT")
    pipe = get_fingpt_pipeline()
    df['fingpt_pred'] = df['sentence'].apply(lambda x: query_fingpt(pipe, x))
    evaluate(df.rename(columns={'fingpt_pred': 'pred'}))
