import pandas as pd
import os

os.makedirs('data/final', exist_ok=True)

df_symptom = pd.read_csv(
    'data/processed/symptom2disease_lemmatized.csv',
    encoding='utf-8-sig')
print(f"\n1. Symptom2Disease: {len(df_symptom)} строк")

df_diag = pd.read_csv(
    'data/processed/diagnozes_lemmatized.csv',
    encoding='utf-8-sig')
print(f"\n2. Diagnozes: {len(df_diag)} строк")

final = pd.concat([df_symptom, df_diag], ignore_index=True)

print(f"\n   Symptom2Disease: {len(df_symptom)} строк")
print(f"   Diagnozes: {len(df_diag)} строк")
print(f"   ВСЕГО: {len(final)} строк")

final.to_csv('data/final/final_dataset.csv', index=False, encoding='utf-8-sig')

print("\n Сохранён: data/final/final_dataset.csv")
print(f"   Колонки: {final.columns.tolist()}")
print(f"  Всего строк: {len(final)}")
print(f"  Уникальных диагнозов: {final['label'].nunique()}")
print("Финальный датасет: data/final/final_dataset.csv")
