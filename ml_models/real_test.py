import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

nlp = spacy.load("ru_core_news_sm")


def clean_text(text):
    """Очистка текста от шума"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text):
    """Лемматизация текста с помощью spaCy"""
    if not isinstance(text, str) or not text:
        return ""
    try:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_punct])
    except BaseException:
        return text


df = pd.read_csv('data/final/final_dataset.csv')
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
vectorizer.fit(df['text'])

df_real = pd.read_csv('data/live_examples/real_cases.csv')

le = LabelEncoder()
y_train_enc = le.fit_transform(df['label'])
le_real = LabelEncoder()
le_real.fit(df['label'])
X_vec = vectorizer.transform(df['text'])

models = {
    "LR": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "SVC": LinearSVC(class_weight='balanced', max_iter=2000, random_state=42),
    "NB": MultinomialNB(alpha=0.1),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "XGB": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

for name, model in models.items():
    if name == "XGB":
        model.fit(X_vec, y_train_enc)
    else:
        model.fit(X_vec, df['label'])

print("Проверка на реальных текстах:\n")
results_real = []
for name, model in models.items():
    correct = 0
    for _, row in df_real.iterrows():
        text = lemmatize_text(clean_text(row['text']))
        X_input = vectorizer.transform([text])
        pred = model.predict(X_input)[0]
        if name == "XGB":
            pred = le_real.inverse_transform([pred])[0]
        if pred == row['label']:
            correct += 1
    acc = round(correct / len(df_real), 4)
    results_real.append({'Model': name, 'Real Accuracy': acc})
    print(f"{name}: {correct}/{len(df_real)} ({acc})")

pd.DataFrame(results_real).to_csv('results/real_results.csv', index=False)
print("\nСохранено в results/real_results.csv")
