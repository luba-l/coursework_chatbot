import pandas as pd
import time
import os

os.makedirs('results/tables', exist_ok=True)

df_symptom = pd.read_csv(
    'data/processed/symptom2disease_clean.csv',
    encoding='utf-8-sig')
print(f"\nSymptom2Disease: {len(df_symptom)} строк")

df_diag = pd.read_csv(
    'data/processed/diagnozes_clean.csv',
    encoding='utf-8-sig')
print(f"Diagnozes: {len(df_diag)} строк")

df_combined = pd.concat([df_symptom, df_diag], ignore_index=True)
sample_texts = df_combined['text'].dropna().tolist()[:200]

print(f"\n Тестовая выборка: {len(sample_texts)} текстов (из Symptom2Disease + Diagnozes)")

try:
    from pymystem3 import Mystem
    mystem = Mystem()
    mystem_available = True
    print("   pymystem3 готов")
except ImportError:
    mystem_available = False
    print("   pymystem3 не установлен")

try:
    import spacy
    nlp_spacy = spacy.load("ru_core_news_sm")
    spacy_available = True
    print("   spaCy готов")
except ImportError:
    spacy_available = False
    print("   spaCy не установлен")
except OSError:
    spacy_available = False
    print("   Модель spaCy не загружена")


def lemmatize_mystem(text):
    if not isinstance(text, str) or not text:
        return ""
    try:
        lemmas = mystem.lemmatize(text)
        return ' '.join([l for l in lemmas if l.strip()])
    except BaseException:
        return text


def lemmatize_spacy(text):
    if not isinstance(text, str) or not text:
        return ""
    try:
        doc = nlp_spacy(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_punct])
    except BaseException:
        return text


results = []

if mystem_available:
    print("\n pymystem3")
    start = time.time()
    lemmatized = [lemmatize_mystem(t) for t in sample_texts]
    elapsed = time.time() - start
    results.append({
        'Лемматизатор': 'pymystem3',
        'Время (сек)': round(elapsed, 2),
        'Скорость (текстов/сек)': round(len(sample_texts) / elapsed, 1)
    })
    print(f"   Время: {elapsed:.2f} сек")
    print(f"   Скорость: {len(sample_texts) / elapsed:.1f} текстов/сек")
    print(f"   Пример исходный: {sample_texts[0][:80]}...")
    print(f"   Пример лемматиз.: {lemmatized[0][:80]}...")

if spacy_available:
    print("\n spaCy")
    start = time.time()
    lemmatized = [lemmatize_spacy(t) for t in sample_texts]
    elapsed = time.time() - start
    results.append({
        'Лемматизатор': 'spaCy',
        'Время (сек)': round(elapsed, 2),
        'Скорость (текстов/сек)': round(len(sample_texts) / elapsed, 1)
    })
    print(f"   Время: {elapsed:.2f} сек")
    print(f"   Скорость: {len(sample_texts) / elapsed:.1f} текстов/сек")
    print(f"   Пример исходный: {sample_texts[0][:80]}...")
    print(f"   Пример лемматиз.: {lemmatized[0][:80]}...")

if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        'results/tables/lemmatizers_comparison.csv',
        index=False,
        encoding='utf-8-sig')

    print(df_results.to_string(index=False))

    fastest = df_results.loc[df_results['Скорость (текстов/сек)'].idxmax()]
    print(
        f"\n Самый быстрый: {
            fastest['Лемматизатор']} ({
            fastest['Скорость (текстов/сек)']} текстов/сек)")

else:
    print("\n Нет доступных лемматизаторов. Установите pymystem3 или spaCy.")
