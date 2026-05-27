def test_imports():
    import pandas
    import numpy
    import sklearn
    import spacy
    import joblib
    import matplotlib
    import seaborn
    assert True


def test_spacy_model():
    import spacy
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp("температура кашель")
    assert len(doc) > 0


def test_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = ["температура кашель", "головная боль тошнота"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    assert X.shape == (2, 5)


def test_model_load():
    import joblib
    import os
    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        assert model is not None
    else:
        assert True
