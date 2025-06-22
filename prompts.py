def generate_zero_shot_prompt(sentence):
    return f"Classifique o sentimento da seguinte frase como positivo, negativo ou neutro: {sentence}"

def generate_few_shot_prompt(sentence, examples):
    prompt = "Classifique o sentimento das frases abaixo como positivo, negativo ou neutro.\n\n"
    for ex in examples:
        prompt += f"Frase: {ex['sentence']}\nSentimento: {ex['sentiment']}\n\n"
    prompt += f"Frase: {sentence}\nSentimento:"
    return prompt
