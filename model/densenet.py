import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np

class DenseNet:
    def __init__(self, task_type='regression', hidden_dims=[512, 256, 128, 64], 
                 dropout_rate=0.3, batch_norm=True, use_skip_connections=True,
                 l1_reg=0.0, l2_reg=0.001, activation='relu'):
        """
        Dense Neural Network for tabular data
        
        Args:
            task_type: 'regression' or 'classification'
            hidden_dims: list of hidden layer dimensions
            dropout_rate: dropout rate
            batch_norm: whether to use batch normalization
            use_skip_connections: whether to use skip connections
            l1_reg: L1 regularization coefficient
            l2_reg: L2 regularization coefficient
            activation: activation function ('relu', 'elu', 'swish')
        """
        self.task_type = task_type
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_skip_connections = use_skip_connections
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activation = activation
        self.model = None

    def get_activation(self):
        """Get activation function"""
        if self.activation == 'relu':
            return tf.nn.relu
        elif self.activation == 'elu':
            return tf.nn.elu
        elif self.activation == 'swish':
            return tf.nn.swish
        else:
            return tf.nn.relu

    def dense_block(self, x, units, layer_idx, use_skip=False, skip_input=None):
        """Dense block with batch norm and dropout"""
        # Dense layer
        dense_out = layers.Dense(
            units,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f"dense_{layer_idx}"
        )(x)
        
        # Batch normalization
        if self.batch_norm:
            dense_out = layers.BatchNormalization(name=f"bn_{layer_idx}")(dense_out)
        
        # Activation
        activation_func = self.get_activation()
        dense_out = layers.Activation(activation_func, name=f"activation_{layer_idx}")(dense_out)
        
        # Skip connection
        if use_skip and skip_input is not None:
            # Ensure dimensions match for skip connection
            if skip_input.shape[-1] != units:
                skip_input = layers.Dense(units, name=f"skip_projection_{layer_idx}")(skip_input)
            dense_out = layers.Add(name=f"skip_add_{layer_idx}")([dense_out, skip_input])
        
        # Dropout
        if self.dropout_rate > 0:
            dense_out = layers.Dropout(self.dropout_rate, name=f"dropout_{layer_idx}")(dense_out)
        
        return dense_out

    def build_model(self, input_dim, output_dim, **kwargs):
        """Build DenseNet model"""
        inputs = layers.Input(shape=(input_dim,), name="input")
        
        # Input normalization
        if self.batch_norm:
            x = layers.BatchNormalization(name="input_bn")(inputs)
        else:
            x = inputs
        
        # Store for skip connections
        skip_connections = []
        
        # Build hidden layers
        for i, units in enumerate(self.hidden_dims):
            # Determine if we should use skip connection
            use_skip = (self.use_skip_connections and 
                       len(skip_connections) > 0 and 
                       i % 2 == 1)  # Skip every other layer
            
            skip_input = skip_connections[-1] if use_skip else None
            
            x = self.dense_block(x, units, i, use_skip, skip_input)
            
            # Store output for potential skip connection
            skip_connections.append(x)
            
            # Keep only last 2 for memory efficiency
            if len(skip_connections) > 2:
                skip_connections.pop(0)
        
        # Final output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(
                output_dim,
                kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name="output"
            )(x)
        else:  # classification
            outputs = layers.Dense(
                output_dim,
                activation='softmax',
                kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name="output"
            )(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="DenseNet")
        
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

    def summary(self):
        """Print model summary"""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")