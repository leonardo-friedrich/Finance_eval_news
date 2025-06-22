from transformers import pipeline

def get_fingpt_pipeline():
    return pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment")

def query_fingpt(pipe, sentence):
    return pipe(sentence)[0]['label'].lower()
