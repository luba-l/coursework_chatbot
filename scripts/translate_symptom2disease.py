import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/raw/symptom2disease.csv', encoding='utf-8')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Колонки: {df.columns.tolist()}")
print(f"Всего строк: {len(df)}")

from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='en', target='ru')

def translate_text(text):
    if not isinstance(text, str):
        return text
    try:
        return translator.translate(text)
    except:
        return text

print("\n Перевод симптомов")
tqdm.pandas(desc="Симптомы")
df['symptom_ru'] = df['text'].progress_apply(translate_text)

print("\n Перевод диагнозов")
tqdm.pandas(desc="Диагнозы")
df['label_ru'] = df['label'].progress_apply(translate_text)

df_final = pd.DataFrame({
    'text': df['symptom_ru'],
    'label': df['label_ru']
})
df_final.to_csv('data/raw/symptom2disease_translated_full.csv', index=False, encoding='utf-8-sig')
print(f"\n Сохранено {len(df_final)} строк")
print(f"Пример: {df_final['text'].iloc[0][:100]}... → {df_final['label'].iloc[0]}")