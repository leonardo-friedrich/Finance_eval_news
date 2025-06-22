# Análise de Sentimento Financeiro

Este projeto implementa análise de sentimento em textos financeiros usando OpenAI GPT e FinGPT.

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Configure a variável de ambiente para a API key do OpenAI:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sua-api-key-aqui"

# Windows Command Prompt
set OPENAI_API_KEY=sua-api-key-aqui

# Linux/Mac
export OPENAI_API_KEY="sua-api-key-aqui"
```

## Uso

Execute o script principal:
```bash
python main.py
```

## Estrutura do Projeto

- `main.py`: Script principal que executa os experimentos
- `inference.py`: Funções para interagir com OpenAI e FinGPT
- `load_data.py`: Carregamento do dataset FinancialPhraseBank
- `prompts.py`: Geração de prompts para o OpenAI
- `FinancialPhraseBank.csv`: Dataset com frases financeiras e seus sentimentos

## Funcionalidades

- **Zero-shot learning**: Classificação usando apenas o prompt
- **Few-shot learning**: Classificação usando exemplos
- **FinGPT**: Modelo especializado em análise financeira
- **Métricas**: Relatório de classificação com precisão, recall e F1-score 