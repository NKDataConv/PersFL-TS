import tensorflow as tf

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=input_shape), # Hidden Units = 32
        tf.keras.layers.Dense(1) # Output Dimension = 1
    ])
    return model