import pandas as pd
import re

with open('data/raw/diagnozes.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Загружено {len(lines)} строк")

data = []
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue

    if i == 0 and 'симптомы' in line.lower():
        continue

    last_semicolon = line.rfind(';')
    if last_semicolon == -1:
        print(f"Строка {i + 1} не содержит разделитель")
        continue
    symptoms = line[:last_semicolon].strip()
    diagnosis = line[last_semicolon + 1:].strip()

    symptoms = re.sub(r';', ',', symptoms)
    symptoms = re.sub(r'\s+', ' ', symptoms)
    data.append({'text': symptoms, 'label': diagnosis})

df = pd.DataFrame(data)
print("\nОбработано {len(df)} строк")

df.to_csv('data/raw/diagnozes_fixed.csv', index=False, encoding='utf-8-sig')
print("\n Сохранено: data/raw/diagnozes_fixed.csv")
print("\n Примеры:")
print(df.head(3).to_string())
