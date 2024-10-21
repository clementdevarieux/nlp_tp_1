import pandas as pd
import numpy as np

def split_dataset(input_filename, split_percentage, output_filename):

    df = pd.read_csv(input_filename)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= split_percentage

    train = df[msk].drop(columns=['split'])
    validation = df[~msk].drop(columns=['split'])

    train.to_csv(output_filename + 'train-test.csv', index=False)
    validation.to_csv(output_filename + 'validation-test.csv', index=False)