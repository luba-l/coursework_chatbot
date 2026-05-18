import pandas as pd
import re
import os

os.makedirs('data/processed', exist_ok=True)

def clean_text(text):
    """Очистка текста от шума"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_label(label):
    """Очистка диагноза"""
    if not isinstance(label, str):
        return ""
    label = label.lower()
    label = re.sub(r'\s+', ' ', label).strip()
    return label

print("\n1. Очистка Symptom2Disease")
df = pd.read_csv('data/raw/symptom2disease_translated_full.csv', encoding='utf-8')
print(f"   Загружено {len(df)} строк")

df['text_cleaned'] = df['text'].apply(clean_text)
df['label_cleaned'] = df['label'].apply(clean_label)

df_clean = df[['text_cleaned', 'label_cleaned']].dropna()
df_clean = df_clean[df_clean['text_cleaned'] != ""]
df_clean = df_clean.rename(columns={'text_cleaned': 'text', 'label_cleaned': 'label'})

df_clean.to_csv('data/processed/symptom2disease_clean.csv', index=False, encoding='utf-8-sig')
print(f"   Сохранено {len(df_clean)} строк в data/processed/symptom2disease_clean.csv")
print(f"   Пример:\n{df_clean.head(2).to_string()}")

print("\n2. Очистка Diagnozes")
df = pd.read_csv('data/raw/diagnozes_fixed.csv', encoding='utf-8-sig')
print(f"   Загружено {len(df)} строк")

df['text_cleaned'] = df['text'].apply(clean_text)
df['label_cleaned'] = df['label'].apply(clean_label)

df_clean = df[['text_cleaned', 'label_cleaned']].dropna()
df_clean = df_clean[df_clean['text_cleaned'] != ""]
df_clean = df_clean.rename(columns={'text_cleaned': 'text', 'label_cleaned': 'label'})

df_clean.to_csv('data/processed/diagnozes_clean.csv', index=False, encoding='utf-8-sig')
print(f"   Сохранено {len(df_clean)} строк в data/processed/diagnozes_clean.csv")
print(f"   Пример:\n{df_clean.head(2).to_string()}")
print("  data/processed/symptom2disease_clean.csv")
print("  data/processed/diagnozes_clean.csv")