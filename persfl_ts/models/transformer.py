import tensorflow as tf

def create_transformer_model(input_shape, num_heads=2, ff_dim=32, num_layers=1):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Transformer Encoder
    x = inputs
    for _ in range(num_layers):
        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward Network
        ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
        ffn_output = tf.keras.layers.Dense(input_shape[-1])(ffn_output)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model 