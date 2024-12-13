import pandas as pd
import numpy as np

def load_timeseries_data(file_path):
    df = pd.read_csv(file_path)
    values = df['value'].values.astype('float32')
    # Reshape für LSTM (Samples, Timesteps, Features)
    values = np.reshape(values, (len(values), 1, 1))  # Annahme: 1 Feature
    return values