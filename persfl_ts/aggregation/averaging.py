import tensorflow as tf
import numpy as np


def federated_averaging(models):
    """Aggregiert die Modelle durch Mittelwertbildung der Gewichte."""
    if not models:
        return None

    # Leeres Modell erstellen
    avg_model = tf.keras.models.clone_model(models[0])

    avg_weights = []

    for layer_index in range(len(models[0].layers)):
        layer_weights = []
        for model in models:
            try:
                layer_weights.append(model.layers[layer_index].get_weights())
            except:
                continue

        if layer_weights:  # Nur Layer betrachten die auch Gewichte haben
            layer_avg_weights = []
            for weight_index in range(len(layer_weights[0])):
                layer_avg_weights.append(np.mean([layer[weight_index] for layer in layer_weights], axis=0))
            avg_weights.append(layer_avg_weights)

    # Gewichte setzen
    weight_index = 0
    for layer_index in range(len(avg_model.layers)):
        try:
            avg_model.layers[layer_index].set_weights(avg_weights[weight_index])
            weight_index += 1
        except:
            continue

    return avg_model