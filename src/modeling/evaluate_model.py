import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import nltk
import joblib

nltk.download('stopwords')

def evaluate(input_filename, model_dump_filename):
    data = pd.read_csv(input_filename)
    X = data['video_name']
    y = data['is_comic']

    # pipeline = Pipeline([
    #     ('vectorizer', CountVectorizer(stop_words=stopwords.words('french'), lowercase=False)),
    #     ('classifier', RandomForestClassifier())
    # ])

    pipeline = joblib.load(model_dump_filename)

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f'Cross-validation scores: {scores}')
    print(f'Mean accuracy: {scores.mean()}')