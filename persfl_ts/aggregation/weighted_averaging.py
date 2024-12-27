import tensorflow as tf
import numpy as np

def weighted_federated_averaging(models, client_data_sizes):
    """Aggregates models by computing a weighted average of the weights based on client data sizes."""
    if not models or not client_data_sizes:
        return None

    # Create an empty model
    avg_model = tf.keras.models.clone_model(models[0])

    avg_weights = []

    for layer_index in range(len(models[0].layers)):
        layer_weights = []
        for model, data_size in zip(models, client_data_sizes):
            try:
                layer_weights.append((model.layers[layer_index].get_weights(), data_size))
            except:
                continue

        if layer_weights:  # Only consider layers that have weights
            layer_avg_weights = []
            for weight_index in range(len(layer_weights[0][0])):
                weighted_sum = np.sum([layer[0][weight_index] * layer[1] for layer in layer_weights], axis=0)
                total_size = np.sum([layer[1] for layer in layer_weights])
                layer_avg_weights.append(weighted_sum / total_size)
            avg_weights.append(layer_avg_weights)

    # Set weights
    weight_index = 0
    for layer_index in range(len(avg_model.layers)):
        try:
            avg_model.layers[layer_index].set_weights(avg_weights[weight_index])
            weight_index += 1
        except:
            continue

    return avg_model 