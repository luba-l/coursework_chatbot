import pandas as pd
import spacy
import os
from tqdm import tqdm

os.makedirs('data/processed', exist_ok=True)

print("\n Загрузка модели spaCy")
nlp = spacy.load("ru_core_news_sm")
print("    Модель загружена")


def lemmatize_text(text):
    """Лемматизация текста с помощью spaCy"""
    if not isinstance(text, str) or not text:
        return ""
    try:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_punct])
    except BaseException:
        return text


print("\n1. Лемматизация Symptom2Disease")
df = pd.read_csv(
    'data/processed/symptom2disease_clean.csv',
    encoding='utf-8-sig')
print(f"   Загружено {len(df)} строк")

tqdm.pandas(desc="Лемматизация Symptom2Disease")
df['text_lemmatized'] = df['text'].progress_apply(lemmatize_text)

df_lemmatized = df[['text_lemmatized', 'label']].rename(
    columns={'text_lemmatized': 'text'})
df_lemmatized.to_csv(
    'data/processed/symptom2disease_lemmatized.csv',
    index=False,
    encoding='utf-8-sig')
print(
    f"   Сохранено {
        len(df_lemmatized)} строк в data/processed/symptom2disease_lemmatized.csv")
print(f"   Пример:\n     Было: {df['text'].iloc[0][:80]}...")
print(f"     Стало: {df_lemmatized['text'].iloc[0][:80]}...")

print("\n2. Лемматизация Diagnozes")
df = pd.read_csv('data/processed/diagnozes_clean.csv', encoding='utf-8-sig')
print(f"   Загружено {len(df)} строк")

tqdm.pandas(desc="Лемматизация Diagnozes")
df['text_lemmatized'] = df['text'].progress_apply(lemmatize_text)

df_lemmatized = df[['text_lemmatized', 'label']].rename(
    columns={'text_lemmatized': 'text'})
df_lemmatized.to_csv(
    'data/processed/diagnozes_lemmatized.csv',
    index=False,
    encoding='utf-8-sig')
print(
    f"   Сохранено {
        len(df_lemmatized)} строк в data/processed/diagnozes_lemmatized.csv")
print(f"   Пример:\n     Было: {df['text'].iloc[0][:80]}...")
print(f"     Стало: {df_lemmatized['text'].iloc[0][:80]}...")

print("  data/processed/symptom2disease_lemmatized.csv")
print("  data/processed/diagnozes_lemmatized.csv")
