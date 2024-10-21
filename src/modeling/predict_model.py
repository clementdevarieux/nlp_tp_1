import joblib
import pandas as pd

def predict(input_filename, model_dump_filename, output_filename):
    pipeline = joblib.load(model_dump_filename)

    new_data = pd.read_csv(input_filename)
    new_titles = new_data['video_name']

    predictions = pipeline.predict(new_titles)

    new_data['predictions'] = predictions
    new_data.to_csv(output_filename, index=False)