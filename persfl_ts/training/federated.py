import tensorflow as tf
from persfl_ts.aggregation.averaging import federated_averaging


def federated_training(clients_data, model, learning_rate, epochs, batch_size):
    num_clients = len(clients_data)
    client_models = [tf.keras.models.clone_model(model) for _ in range(num_clients)]

    for client_model in client_models:
        client_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    for epoch in range(epochs):
        for client_id, client_data in enumerate(clients_data):
            client_models[client_id].fit(client_data, np.random.rand(*client_data.shape[:2], 1), epochs=1,
                                         batch_size=batch_size, verbose=0)  # Hier sollte ein sinnvolles Target sein

        global_model = federated_averaging(client_models)

        for client_model in client_models:
            client_model.set_weights(global_model.get_weights())

        print(f"Epoche {epoch + 1}/{epochs}")

    return global_model