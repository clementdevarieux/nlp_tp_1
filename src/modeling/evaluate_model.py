import pandas as pd
from sklearn.model_selection import cross_val_score
import nltk
import joblib

nltk.download('stopwords')

def evaluate(input_filename, model_dump_filename):
    data = pd.read_csv(input_filename)
    X = data['video_name']
    y = data['is_comic']

    pipeline = joblib.load(model_dump_filename)

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f'Cross-validation scores: {scores}')
    print(f'Mean accuracy: {scores.mean()}')