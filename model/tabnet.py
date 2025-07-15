import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TabNet:
    def __init__(self, task_type='regression', n_d=64, n_a=64, n_steps=5, gamma=1.3, 
                 n_independent=2, n_shared=2, epsilon=1e-15):
        """
        TabNet model for tabular data
        
        Args:
            task_type: 'regression' or 'classification'
            n_d: width of the decision prediction layer
            n_a: width of the attention embedding for each mask
            n_steps: number of decision steps
            gamma: coefficient for feature reusage in the masks
            n_independent: number of independent GLU layers at each step
            n_shared: number of shared GLU layers at each step
            epsilon: avoid log(0)
        """
        self.task_type = task_type
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.model = None

    def glu_block(self, x, n_units, shared_layers=None, name_prefix=""):
        """Gated Linear Unit block"""
        if shared_layers is not None:
            x = shared_layers(x)
        
        x = layers.Dense(n_units * 2, name=f"{name_prefix}_dense")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        
        # Split and apply GLU using Lambda layer with explicit output_shape
        def glu_activation(x):
            gate = tf.nn.sigmoid(x[:, :n_units])
            output = x[:, n_units:] * gate
            return output
        
        output = layers.Lambda(
            glu_activation, 
            output_shape=(n_units,),
            name=f"{name_prefix}_glu"
        )(x)
        
        return output

    def attentive_transformer(self, x, prior, name_prefix=""):
        """Attentive transformer for feature selection"""
        x = layers.Dense(self.n_a, name=f"{name_prefix}_dense")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        
        # Apply prior using Multiply layer instead of Lambda
        x = layers.Multiply(name=f"{name_prefix}_prior_apply")([x, prior])
        
        # Sparsemax activation (approximated with softmax) using Activation layer
        mask = layers.Activation('softmax', name=f"{name_prefix}_softmax")(x)
        
        return mask

    def feature_transformer(self, x, shared_layers, step, name_prefix=""):
        """Feature transformer block"""
        # Shared layers
        if shared_layers is not None:
            x = shared_layers(x)
        
        # Independent layers for this step
        for i in range(self.n_independent):
            x = self.glu_block(x, self.n_d + self.n_a, 
                             name_prefix=f"{name_prefix}_independent_{i}")
        
        return x

    def build_model(self, input_dim, output_dim, **kwargs):
        """Build TabNet model"""
        inputs = layers.Input(shape=(input_dim,), name="input")
        
        # Initial batch norm
        x = layers.BatchNormalization(name="initial_bn")(inputs)
        
        # Shared layers
        shared_layers = []
        for i in range(self.n_shared):
            shared_layers.append(
                layers.Dense(self.n_d + self.n_a, activation='relu',
                           name=f"shared_{i}")
            )
        
        # Initialize prior - same shape as input features
        prior = layers.Lambda(lambda x: tf.ones_like(x), name="prior_init")(inputs)
        decision_outputs = []
        
        for step in range(self.n_steps):
            # Feature selection mask - output should match input features shape
            mask = self.attentive_transformer_v2(
                inputs, prior, name_prefix=f"step_{step}_attention"
            )
            
            # Apply mask to original input features
            masked_features = layers.Multiply(name=f"step_{step}_mask_apply")([inputs, mask])
            
            # Feature transformation
            transformed = masked_features
            for shared_layer in shared_layers:
                transformed = shared_layer(transformed)
                transformed = layers.BatchNormalization()(transformed)
                transformed = layers.Activation('relu')(transformed)
            
            # Split into decision and attention parts
            decision_out = layers.Dense(self.n_d, activation='relu',
                                      name=f"step_{step}_decision")(transformed)
            
            decision_outputs.append(decision_out)
            
            # Update prior for next step
            prior = layers.Lambda(
                lambda inputs: inputs[0] * (self.gamma - inputs[1]),
                output_shape=lambda input_shape: input_shape[0],
                name=f"step_{step}_prior_update"
            )([prior, mask])
        
        # Aggregate decision outputs using Add layers
        if len(decision_outputs) == 1:
            decision_sum = decision_outputs[0]
        else:
            decision_sum = layers.Add(name="decision_aggregation")(decision_outputs)
        
        # Final output layer
        if self.task_type == 'regression':
            outputs = layers.Dense(output_dim, name="output")(decision_sum)
        else:  # classification
            outputs = layers.Dense(output_dim, activation='softmax', 
                                 name="output")(decision_sum)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="TabNet")
        
        # Compile model
        if self.task_type == 'regression':
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mse',
                metrics=['mae', 'mse']
            )
        else:  # classification
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model

    def attentive_transformer_v2(self, features, prior, name_prefix=""):
        """Attentive transformer that outputs mask with same shape as input features"""
        # Project to attention space
        attention = layers.Dense(self.n_a, name=f"{name_prefix}_dense")(features)
        attention = layers.BatchNormalization(name=f"{name_prefix}_bn")(attention)
        
        # Project back to feature space for mask
        mask_logits = layers.Dense(features.shape[-1], name=f"{name_prefix}_mask_proj")(attention)
        
        # Apply prior
        mask_logits = layers.Multiply(name=f"{name_prefix}_prior_apply")([mask_logits, prior])
        
        # Softmax to get attention mask
        mask = layers.Activation('softmax', name=f"{name_prefix}_softmax")(mask_logits)
        
        return mask

    def get_model(self):
        """Return compiled model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model