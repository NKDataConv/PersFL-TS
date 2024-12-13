import tensorflow as tf
import numpy as np
from persfl_ts.models.lstm import create_lstm_model
from persfl_ts.datasets.timeseries import load_timeseries_data
from persfl_ts.training.federated import federated_training

# Hyperparameter
input_shape = (1, 1) # Timesteps, Features
learning_rate = 0.01
epochs = 10
batch_size = 32

# Daten laden
client1_data = load_timeseries_data("data/client_1.csv")
client2_data = load_timeseries_data("data/client_2.csv")
clients_data = [client1_data, client2_data]

# Modell erstellen
model = create_lstm_model(input_shape)

# Federated Training
global_model = federated_training(clients_data, model, learning_rate, epochs, batch_size)

print("Federated Training abgeschlossen.")

# Modell speichern (optional)
global_model.save('federated_model')