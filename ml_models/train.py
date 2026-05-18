import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/final/final_dataset.csv')

TFIDF_PARAMS = {'min_df': 2, 'max_df': 0.95, 'ngram_range': (1, 2)}
X_train_text, X_test_text, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.15, stratify=df['label'], random_state=42)
vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

svc = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
svc.fit(X_train, y_train)

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

le = LabelEncoder()
le.fit(df['label'])
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)
y_enc = le.transform(df['label'])
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train_enc)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
models = {
    'LR': (lr, LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    'SVC': (svc, LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)),
    'NB': (nb, MultinomialNB(alpha=0.1)),
    'MLP': (mlp, MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
    'XGB': (xgb, XGBClassifier(eval_metric='mlogloss', random_state=42))
}
for name, (trained_model, clean_model) in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf', clean_model)
    ])
    if name == 'XGB':
        scores = cross_val_score(pipeline, df['text'], y_enc, cv=skf, scoring='balanced_accuracy')
        train_score = balanced_accuracy_score(y_train_enc, trained_model.predict(X_train))
        test_score = balanced_accuracy_score(y_test_enc, trained_model.predict(X_test))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, trained_model.predict(X_test), average='macro', zero_division=0)
    else:
        scores = cross_val_score(pipeline, df['text'], df['label'], cv=skf, scoring='balanced_accuracy')
        train_score = balanced_accuracy_score(y_train, trained_model.predict(X_train))
        test_score = balanced_accuracy_score(y_test, trained_model.predict(X_test))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, trained_model.predict(X_test), average='macro', zero_division=0)
    results.append({
        'Model': name,
        'Train BA': round(train_score, 4),
        'Test BA': round(test_score, 4),
        'CV Mean': round(scores.mean(), 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-score': round(f1, 4)
    })
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('results/train_results.csv', index=False)
print("Сохранено в results/train_results.csv")