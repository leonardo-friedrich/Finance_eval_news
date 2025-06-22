import openai
from transformers import pipeline
import os

client = openai.OpenAI(api_key=os.getenv("") or "")

def get_fingpt_pipeline():
    return pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment")

def query_fingpt(pipe, sentence):
    return pipe(sentence)[0]['label'].lower()

def query_openai(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()