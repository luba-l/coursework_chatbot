import pandas as pd

df = pd.read_csv('data/raw/RuMedPrimeData.tsv', sep='\t', encoding='utf-8')

df_unified = pd.DataFrame({
    'text': df['symptoms'],
    'label': df['icd10']
})

df_unified.to_csv('data/raw/rumed_unified.csv', index=False, encoding='utf-8-sig')
print(f" Сохранено: data/raw/rumed_unified.csv")
print(f"   Пример:\n{df_unified.head(2)}")