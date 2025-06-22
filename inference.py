import openai
from transformers import pipeline
import os


def get_fingpt_pipeline():
    return pipeline("text-classification", model="ProsusAI/finbert")


def query_fingpt(pipe, sentence):
    return pipe(sentence)[0]['label'].lower()

def extract_sentiment_label(text: str) -> str:
    text = text.lower()
    if "positivo" in text:
        return "positive"
    elif "negativo" in text:
        return "negative"
    else:
        return "neutral"

def query_openai(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_output = response.choices[0].message.content.strip().lower()
    return extract_sentiment_label(raw_output)