import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

df = pd.read_csv('data/final/final_dataset.csv', encoding='utf-8-sig')
print(f"\nЗагружено {len(df)} строк")
print(f"Колонки: {df.columns.tolist()}")
print(f"Уникальных диагнозов: {df['label'].nunique()}")

print("\n1. Анализ длины текстов")
lengths = df['text'].fillna('').astype(str).str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(lengths, bins=30, edgecolor='black', color='skyblue')
axes[0].set_xlabel('Длина текста (символы)')
axes[0].set_ylabel('Частота')
axes[0].set_title('Финальный датасет - Гистограмма')
axes[0].axvline(lengths.mean(), color='red', linestyle='--', label=f'Средняя: {lengths.mean():.0f}')
axes[0].legend()

axes[1].boxplot(lengths, vert=False)
axes[1].set_xlabel('Длина текста (символы)')
axes[1].set_title('Финальный датасет - Boxplot')

plt.tight_layout()
plt.savefig('results/figures/final_dataset_length.png', dpi=150)
plt.close()

print(f"   Средняя длина: {lengths.mean():.1f}")
print(f"   Медиана: {lengths.median():.1f}")
print(f"   Мин: {lengths.min()}, Макс: {lengths.max()}")
print("\n2. Топ-20 диагнозов")
top_labels = df['label'].value_counts().head(20)

plt.figure(figsize=(12, 8))
top_labels.plot(kind='barh', color='green', edgecolor='black')
plt.xlabel('Количество')
plt.title('Финальный датасет - Топ-20 диагнозов')
plt.tight_layout()
plt.savefig('results/figures/final_dataset_top_diagnoses.png', dpi=150)
plt.close()

top_labels.to_csv(
    'results/tables/final_dataset_top_diagnoses.csv',
    encoding='utf-8-sig')
print(f"   Самый частый диагноз: {top_labels.index[0]} ({top_labels.iloc[0]} раз)")
print(f"   Топ-5: {', '.join([f'{top_labels.index[i]} ({top_labels.iloc[i]})' for i in range(min(5, len(top_labels)))])}")

print("\n3. Генерация облака слов")
all_text = ' '.join(df['text'].fillna('').astype(str))

wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='viridis',
    max_words=100,
    contour_width=1,
    contour_color='steelblue'
).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Финальный датасет - Облако слов')
plt.tight_layout()
plt.savefig('results/figures/final_dataset_wordcloud.png', dpi=150)
plt.close()
print("   Облако слов сохранено")

print("\n4. Топ-20 слов")
words = ' '.join(df['text'].fillna('').astype(str)).lower().split()
word_counts = Counter(words)
top_words = word_counts.most_common(20)

words_list, counts_list = zip(*top_words)

plt.figure(figsize=(10, 6))
plt.barh(words_list[::-1], counts_list[::-1], color='coral')
plt.xlabel('Частота')
plt.title('Финальный датасет - Топ-20 слов')
plt.tight_layout()
plt.savefig('results/figures/final_dataset_top_words.png', dpi=150)
plt.close()

print("   Топ-5 слов:")
for i, (word, count) in enumerate(top_words[:5]):
    print(f"     {i + 1}. '{word}' — {count} раз")

stats = pd.DataFrame({
    'Показатель': [
        'Всего строк',
        'Уникальных диагнозов',
        'Средняя длина текста',
        'Медианная длина текста',
        'Мин. длина',
        'Макс. длина',
        'Стандартное отклонение'
    ],
    'Значение': [
        len(df),
        df['label'].nunique(),
        round(lengths.mean(), 1),
        round(lengths.median(), 1),
        lengths.min(),
        lengths.max(),
        round(lengths.std(), 1)
    ]
})
stats.to_csv('results/tables/final_dataset_stats.csv',
             index=False, encoding='utf-8-sig')
