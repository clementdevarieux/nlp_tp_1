# evaluate_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Charger les données
data = pd.read_csv('src/data/names_train.csv')

# Séparer les features et les labels
X = data['title']
y = data['label']

# Créer le pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

# Évaluer le modèle avec la cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {scores.mean()}')