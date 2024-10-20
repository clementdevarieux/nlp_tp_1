# predict_model.py
import joblib
import pandas as pd

def predict(input_filename, model_dump_filename, output_filename):
    # Charger le modèle
    pipeline = joblib.load(model_dump_filename)

    # Charger les nouveaux titres de vidéos
    new_data = pd.read_csv(input_filename)
    new_titles = new_data['title']

    # Faire des prédictions
    predictions = pipeline.predict(new_titles)

    # Ajouter les prédictions au DataFrame et sauvegarder
    new_data['predictions'] = predictions
    new_data.to_csv(output_filename, index=False)