import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.d_k = input_dim
        self.d_v = input_dim
        
        # Learnable weight matrices for query, key, and value projections
        self.W_Q = tf.keras.layers.Dense(self.d_k)
        self.W_K = tf.keras.layers.Dense(self.d_k)
        self.W_V = tf.keras.layers.Dense(self.d_v)
        
        # Optional projection layer to map back to input dimension
        self.W_O = tf.keras.layers.Dense(input_dim)
    
    def call(self, X):
        # X shape: (B, T, input_dim)

        # Compute queries, keys, and values
        Q = self.W_Q(X)  # Shape: (B, T, d_k)
        K = self.W_K(X)  # Shape: (B, T, d_k)
        V = self.W_V(X)  # Shape: (B, T, d_v)
        
        # Compute scaled dot-product attention scores
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32))  # Shape: (B, T, T)
        attention_weights = tf.nn.softmax(scores, axis=-1)  # Shape: (B, T, T)
        
        # Apply attention weights to values
        weighted_values = tf.matmul(attention_weights, V)  # Shape: (B, T, d_v)
        
        # Optionally project the output back to the input dimension
        output = self.W_O(weighted_values)  # Shape: (B, T, input_dim)
        
        return output