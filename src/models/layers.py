"""
Custom layers for the BDS-3 MEO satellite yaw attitude analysis model
"""

import tensorflow as tf
import numpy as np

class EclipseAttention(tf.keras.layers.Layer):
    """
    Custom attention layer that gives higher weights to eclipse periods
    Eclipse period is defined as when |β| < 12.9°
    """
    
    def __init__(self, units):
        """
        Initialize the Eclipse Attention layer
        
        Args:
            units: Number of attention units
        """
        super(EclipseAttention, self).__init__()
        self.attention_dense = tf.keras.layers.Dense(units, activation='tanh')
        self.attention_output = tf.keras.layers.Dense(1, activation=None)
    
    def call(self, inputs, beta=None):
        """
        Forward pass with special attention to eclipse periods
        
        Args:
            inputs: Input tensor
            beta: Sun elevation angle tensor
            
        Returns:
            Attention-weighted output
        """
        # Basic attention mechanism
        attention_features = self.attention_dense(inputs)
        attention_scores = self.attention_output(attention_features)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # If beta is provided, apply eclipse weighting
        if beta is not None:
            # Create eclipse mask: True where |beta| < 12.9°
            eclipse_mask = tf.abs(beta) < 12.9
            eclipse_mask = tf.cast(eclipse_mask, tf.float32)
            
            # Enhance attention weights for eclipse periods (multiply by 3.0)
            eclipse_factor = tf.where(eclipse_mask, 3.0, 1.0)
            attention_weights = attention_weights * tf.expand_dims(eclipse_factor, -1)
            
            # Re-normalize weights
            attention_sum = tf.reduce_sum(attention_weights, axis=1, keepdims=True)
            attention_weights = attention_weights / (attention_sum + tf.keras.backend.epsilon())
        
        # Apply attention weights to input
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

class DynamicsConstraint(tf.keras.layers.Layer):
    """
    Physics constraint layer implementing yaw rate dynamics from equation (2)
    ψ˙n = sin²μ + tan²β / μ˙tanβcosμ
    """
    
    def __init__(self):
        """Initialize the DynamicsConstraint layer"""
        super(DynamicsConstraint, self).__init__()
    
    def call(self, yaw, beta, mu, mu_dot=0.008):
        """
        Apply physics constraint to the yaw prediction
        
        Args:
            yaw: Predicted yaw angles
            beta: Sun elevation angles
            mu: Orbit angles
            mu_dot: Mean motion angular rate (deg/s)
            
        Returns:
            Physics constraint loss
        """
        # Convert to radians
        yaw_rad = tf.cast(yaw, tf.float32) * np.pi / 180.0
        beta_rad = tf.cast(beta, tf.float32) * np.pi / 180.0
        mu_rad = tf.cast(mu, tf.float32) * np.pi / 180.0
        
        # Avoid division by zero
        epsilon = 1e-8
        safe_beta = tf.where(tf.abs(beta_rad) < epsilon, 
                            tf.sign(beta_rad) * epsilon, 
                            beta_rad)
        safe_cos_mu = tf.where(tf.abs(tf.cos(mu_rad)) < epsilon,
                              tf.sign(tf.cos(mu_rad)) * epsilon,
                              tf.cos(mu_rad))
        
        # Calculate the theoretical yaw rate based on equation (2)
        numerator = tf.square(tf.sin(mu_rad)) + tf.square(tf.tan(safe_beta))
        denominator = mu_dot * tf.tan(safe_beta) * safe_cos_mu
        theoretical_yaw_rate = numerator / denominator
        
        # Calculate the actual yaw rate from predicted yaw
        # For this we'd need the time derivative, which requires consecutive timestamps
        # Here we're just using an approximate constraint on yaw values
        
        # For eclipse periods, ensure yaw rate doesn't exceed control system threshold (~0.1°/s)
        eclipse_mask = tf.abs(beta) < 12.9
        eclipse_mask = tf.cast(eclipse_mask, tf.float32)
        
        # Apply constraint for eclipse periods
        # Force yaw to follow nominal pattern in non-eclipse periods
        # For eclipse periods, allow more flexibility
        return theoretical_yaw_rate, eclipse_mask
        
class MultiFrequencyConv(tf.keras.layers.Layer):
    """
    Multi-frequency convolution layer for capturing different periodic patterns
    Implements ECOMC model structure with different frequency components
    """
    
    def __init__(self, frequencies=[1, 2, 4]):
        """
        Initialize the MultiFrequencyConv layer
        
        Args:
            frequencies: List of frequencies to model (1, 2, 4 for ECOMC)
        """
        super(MultiFrequencyConv, self).__init__()
        self.frequencies = frequencies
        self.cos_convs = []
        self.sin_convs = []
        
        for _ in frequencies:
            # Create separate convolutional filters for each frequency
            self.cos_convs.append(tf.keras.layers.Conv1D(16, kernel_size=3, padding='same'))
            self.sin_convs.append(tf.keras.layers.Conv1D(16, kernel_size=3, padding='same'))
    
    def call(self, inputs):
        """
        Apply multi-frequency convolution
        
        Args:
            inputs: Input tensor including orbit angle
            
        Returns:
            Multi-frequency convolutional features
        """
        # Extract orbit angle
        orbit_angle = inputs[:, :, 2:3]  # Assuming orbit angle is the 3rd feature
        
        # Apply frequency transformations
        freq_outputs = []
        
        for i, freq in enumerate(self.frequencies):
            # Create cos and sin features for this frequency
            cos_feature = tf.cos(freq * orbit_angle * np.pi / 180.0)
            sin_feature = tf.sin(freq * orbit_angle * np.pi / 180.0)
            
            # Apply convolutions
            cos_output = self.cos_convs[i](cos_feature)
            sin_output = self.sin_convs[i](sin_feature)
            
            # Combine
            freq_output = tf.concat([cos_output, sin_output], axis=-1)
            freq_outputs.append(freq_output)
        
        # Concatenate all frequency outputs
        multi_freq_output = tf.concat(freq_outputs, axis=-1)
        
        # Combine with original input
        output = tf.concat([inputs, multi_freq_output], axis=-1)
        
        return output
