# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Charger les données
data = pd.read_csv('src/data/names_train.csv')

# Séparer les features et les labels
X = data['title']
y = data['label']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Évaluer le modèle
y_pred = pipeline.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Sauvegarder le modèle
joblib.dump(pipeline, 'model.pkl')