import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('results/train_results.csv')
real_df = pd.read_csv('results/real_results.csv')
final = train_df.merge(real_df, on='Model')
print("Итоговая таблица:")
final.to_csv('results/final_results.csv', index=False)
print(final.to_string(index=False))

best = final.sort_values(
    ['CV Mean', 'Test BA', 'Real Accuracy'],
    ascending=False
).iloc[0]

best_name = best['Model']
print(f"\nЛучшая модель: {best_name}")

df = pd.read_csv('data/final/final_dataset.csv')
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

if best_name == "LR":
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42)
elif best_name == "SVC":
    model = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
elif best_name == "NB":
    model = MultinomialNB(alpha=0.1)
elif best_name == "MLP":
    model = MLPClassifier(
        hidden_layer_sizes=(
            100,
            50),
        max_iter=500,
        random_state=42)
elif best_name == "XGB":
    le = LabelEncoder()
    y = le.fit_transform(y)
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
else:
    raise ValueError("Неизвестная модель")

model.fit(X, y)
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print(f"\nМодель {best_name} сохранена в папке models/")
