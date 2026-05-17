import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/final/final_dataset.csv')

vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

svc = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
svc.fit(X_train, y_train)

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train_enc)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models_all = {"LR": lr, "SVC": svc, "NB": nb, "MLP": mlp, "XGB": xgb}
results = []
for name, model in models_all.items():
    if name == "XGB":
        train_score = balanced_accuracy_score(y_train_enc, model.predict(X_train))
        test_score = balanced_accuracy_score(y_test_enc, model.predict(X_test))
        scores = cross_val_score(model, X, y_enc, cv=skf, scoring='balanced_accuracy')
    else:
        train_score = balanced_accuracy_score(y_train, model.predict(X_train))
        test_score = balanced_accuracy_score(y_test, model.predict(X_test))
        scores = cross_val_score(model, X, y, cv=skf, scoring='balanced_accuracy')
    results.append({'Model': name, 'Train': round(train_score, 4), 'Test': round(test_score, 4), 'CV Mean': round(scores.mean(), 4)})
results_df = pd.DataFrame(results)
print(results_df)