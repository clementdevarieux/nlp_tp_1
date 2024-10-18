# predict_model.py
import joblib
import pandas as pd

# Charger le modèle
pipeline = joblib.load('model.pkl')

# Charger les nouveaux titres de vidéos
new_data = pd.read_csv('src/data/names_predict.csv')
new_titles = new_data['title']

# Faire des prédictions
predictions = pipeline.predict(new_titles)

# Ajouter les prédictions au DataFrame et sauvegarder
new_data['predictions'] = predictions
new_data.to_csv('src/data/names_predictions.csv', index=False)