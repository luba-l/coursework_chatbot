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

df_rumed = pd.read_csv('data/raw/rumed_unified.csv', encoding='utf-8-sig')
print(f"\n1. RuMedPrime: {len(df_rumed)} строк")
print(f"   Колонки: {df_rumed.columns.tolist()}")

df_symptom = pd.read_csv(
    'data/raw/symptom2disease_translated_full.csv',
    encoding='utf-8')
print(f"\n2. Symptom2Disease: {len(df_symptom)} строк")
print(f"   Колонки: {df_symptom.columns.tolist()}")

df_diag = pd.read_csv('data/raw/diagnozes_fixed.csv', encoding='utf-8-sig')
print(f"\n3. Diagnozes: {len(df_diag)} строк")
print(f"   Колонки: {df_diag.columns.tolist()}")


def analyze_lengths(df, text_col, name, filename):
    """Анализ длины текстов: гистограмма, boxplot, статистика"""
    lengths = df[text_col].fillna('').astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(lengths, bins=30, edgecolor='black', color='skyblue')
    axes[0].set_xlabel('Длина текста (символы)')
    axes[0].set_ylabel('Частота')
    axes[0].set_title(f'{name} - Гистограмма')
    axes[0].axvline(
        lengths.mean(),
        color='red',
        linestyle='--',
        label=f'Средняя: {
            lengths.mean():.0f}')
    axes[0].legend()

    axes[1].boxplot(lengths, vert=False)
    axes[1].set_xlabel('Длина текста (символы)')
    axes[1].set_title(f'{name} - Boxplot')

    plt.tight_layout()
    plt.savefig(f'results/figures/{filename}_length.png', dpi=150)
    plt.close()

    return {
        'Датасет': name,
        'Средняя длина': round(lengths.mean(), 1),
        'Медиана': round(lengths.median(), 1),
        'Мин': lengths.min(),
        'Макс': lengths.max(),
        'Стд': round(lengths.std(), 1)
    }


def plot_top_words(df, text_col, name, filename, top_n=15):
    """Топ-N самых частотных слов"""
    all_text = ' '.join(df[text_col].fillna('').astype(str).str.lower())
    words = all_text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)

    words_list, counts_list = zip(*top_words)

    plt.figure(figsize=(10, 6))
    plt.barh(words_list[::-1], counts_list[::-1], color='coral')
    plt.xlabel('Частота')
    plt.title(f'{name} - Топ {top_n} слов')
    plt.tight_layout()
    plt.savefig(f'results/figures/{filename}_topwords.png', dpi=150)
    plt.close()


def plot_wordcloud(df, text_col, name, filename):
    """Облако слов"""
    all_text = ' '.join(df[text_col].fillna('').astype(str))

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f'results/figures/{filename}_wordcloud.png', dpi=150)
    plt.close()


def plot_top_diagnoses(df, label_col, name, filename, top_n=10):
    """Топ диагнозов"""
    top_labels = df[label_col].value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    top_labels.plot(kind='barh', color='green', edgecolor='black')
    plt.xlabel('Количество')
    plt.title(f'{name} - Топ-{top_n} диагнозов')
    plt.tight_layout()
    plt.savefig(f'results/figures/{filename}_diagnoses.png', dpi=150)
    plt.close()

    return top_labels


stats = []

stats.append(analyze_lengths(df_rumed, 'text', 'RuMedPrime', 'rumed_raw'))
plot_top_words(df_rumed, 'text', 'RuMedPrime', 'rumed_raw')
plot_wordcloud(df_rumed, 'text', 'RuMedPrime', 'rumed_raw')
plot_top_diagnoses(df_rumed, 'label', 'RuMedPrime', 'rumed_raw')

stats.append(
    analyze_lengths(
        df_symptom,
        'text',
        'Symptom2Disease',
        'symptom_raw'))
plot_top_words(df_symptom, 'text', 'Symptom2Disease', 'symptom_raw')
plot_wordcloud(df_symptom, 'text', 'Symptom2Disease', 'symptom_raw')
plot_top_diagnoses(df_symptom, 'label', 'Symptom2Disease', 'symptom_raw')

stats.append(analyze_lengths(df_diag, 'text', 'Diagnozes', 'diagnozes_raw'))
plot_top_words(df_diag, 'text', 'Diagnozes', 'diagnozes_raw')
plot_wordcloud(df_diag, 'text', 'Diagnozes', 'diagnozes_raw')
plot_top_diagnoses(df_diag, 'label', 'Diagnozes', 'diagnozes_raw')

df_comparison = pd.DataFrame(stats)
df_comparison = df_comparison.round(1)
df_comparison.to_csv(
    'results/tables/datasets_comparison_raw.csv',
    index=False,
    encoding='utf-8-sig')

print(df_comparison.to_string(index=False))

plt.figure(figsize=(8, 5))
bars = plt.bar(df_comparison['Датасет'], df_comparison['Средняя длина'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Средняя длина текста (символы)')
plt.title('Сравнение средней длины симптомов в датасетах')

for bar, val in zip(bars, df_comparison['Средняя длина']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/avg_length_comparison.png', dpi=150)
plt.close()
