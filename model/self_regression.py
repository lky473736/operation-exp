import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SelfRegression:
    def __init__(self, task_type='regression', d_model=128, num_heads=8, 
                 num_layers=4, dff=512, dropout_rate=0.1):
        """
        Self-Regression model using transformer architecture
        
        Args:
            task_type: 'regression' or 'classification'
            d_model: dimension of model
            num_heads: number of attention heads
            num_layers: number of transformer blocks
            dff: dimension of feed forward network
            dropout_rate: dropout rate
        """
        self.task_type = task_type
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.model = None

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

    def multi_head_attention(self, v, k, q, mask=None, name="mha"):
        """Multi-head attention layer"""
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = layers.Dense(self.d_model, name=f"{name}_q")(q)
        k = layers.Dense(self.d_model, name=f"{name}_k")(k)
        v = layers.Dense(self.d_model, name=f"{name}_v")(v)
        
        # Reshape for multi-head
        depth = self.d_model // self.num_heads
        
        q = tf.reshape(q, [batch_size, -1, self.num_heads, depth])
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        
        k = tf.reshape(k, [batch_size, -1, self.num_heads, depth])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        
        v = tf.reshape(v, [batch_size, -1, self.num_heads, depth])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # Attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                    [batch_size, -1, self.d_model])
        
        # Final linear projection
        output = layers.Dense(self.d_model, name=f"{name}_output")(concat_attention)
        
        return output

    def point_wise_feed_forward_network(self, d_model, dff, name="ffn"):
        """Point wise feed forward network"""
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu', name=f"{name}_dense1"),
            layers.Dense(d_model, name=f"{name}_dense2")
        ])

    def encoder_layer(self, x, training, mask=None, layer_idx=0):
        """Single encoder layer"""
        # Multi-head attention
        attn_output = self.multi_head_attention(
            x, x, x, mask, name=f"layer_{layer_idx}_mha")
        attn_output = layers.Dropout(self.dropout_rate)(attn_output, training=training)
        out1 = layers.LayerNormalization(epsilon=1e-6,
                                       name=f"layer_{layer_idx}_ln1")(x + attn_output)
        
        # Feed forward
        ffn = self.point_wise_feed_forward_network(
            self.d_model, self.dff, name=f"layer_{layer_idx}_ffn")
        ffn_output = ffn(out1)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output, training=training)
        out2 = layers.LayerNormalization(epsilon=1e-6,
                                       name=f"layer_{layer_idx}_ln2")(out1 + ffn_output)
        
        return out2

    def positional_encoding(self, position, d_model):
        """Generate positional encoding"""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        """Calculate angles for positional encoding"""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def build_model(self, input_dim, output_dim, **kwargs):
        """Build Self-Regression model"""
        inputs = layers.Input(shape=(input_dim,), name="input")
        
        # Reshape input to sequence format
        x = tf.expand_dims(inputs, axis=1)  # (batch, 1, features)
        
        # Input embedding
        x = layers.Dense(self.d_model, name="input_embedding")(x)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        pos_encoding = self.positional_encoding(input_dim, self.d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_encoding[:, :seq_len, :]
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Encoder layers
        for i in range(self.num_layers):
            x = self.encoder_layer(x, training=True, layer_idx=i)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # Dropout before final layer
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Final output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(output_dim, name="output")(x)
        else:  # classification
            outputs = layers.Dense(output_dim, activation='softmax', 
                                 name="output")(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="SelfRegression")
        
        # Compile model
        if self.task_type == 'regression':
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mse']
            )
        else:  # classification
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model

    def get_model(self):
        """Return compiled model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model