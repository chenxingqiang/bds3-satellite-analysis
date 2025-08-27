"""
Solar Radiation Pressure (SRP) Compensation Model using Graph Convolution
"""

import tensorflow as tf
import numpy as np
from src.models.layers import MultiFrequencyConv

class SimpleGraphConv(tf.keras.layers.Layer):
    """Simplified graph convolution layer without spektral dependency"""
    
    def __init__(self, filters, activation='relu'):
        super(SimpleGraphConv, self).__init__()
        self.filters = filters
        self.dense = tf.keras.layers.Dense(filters, activation=activation)
        
    def call(self, inputs):
        features, adjacency = inputs
        # Simple message passing: multiply features by adjacency and apply dense layer
        aggregated = tf.matmul(adjacency, features)
        output = self.dense(aggregated)
        return output

class SRPCompensationModel(tf.keras.Model):
    """Graph convolution model for SRP compensation"""
    
    def __init__(self, gcn_filters=64, gcn_kernel=3):
        """
        Initialize the SRP Compensation Model
        
        Args:
            gcn_filters: Number of GCN filters
            gcn_kernel: Kernel size for GCN
        """
        super(SRPCompensationModel, self).__init__()
        
        # Input feature processing
        self.input_dense = tf.keras.layers.Dense(32, activation='relu')
        
        # Multi-frequency convolution layer for capturing periodic patterns
        self.multi_freq_conv = MultiFrequencyConv(frequencies=[1, 2, 4])  # Following ECOMC model
        
        # Graph convolution layer for satellite relationship modeling
        self.gcn = SimpleGraphConv(gcn_filters, activation='relu')
        
        # Output layers for D, Y, B components
        self.d_output = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4)  # D0, Dc, D2c, D4c from simplified ECOMC model
        ])
        
        self.y_output = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2)  # Y0, Ys
        ])
        
        self.b_output = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2)  # B0, Bs
        ])
    
    def build_adjacency(self, batch_size):
        """
        Build adjacency matrix for batch processing
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Adjacency matrix
        """
        # Create simple identity matrix for batch processing
        adj = tf.eye(batch_size)
        return adj
    
    def call(self, inputs):
        """
        Forward pass for the SRP compensation model
        
        Args:
            inputs: Input tensor with shape [batch, time_steps, features]
                   features = [time, beta, mu, sat_type_onehot]
            
        Returns:
            SRP parameters for D, Y, B directions
        """
        # Process input features
        x = self.input_dense(inputs)
        
        # Apply multi-frequency convolution for periodic patterns
        x = self.multi_freq_conv(x)
        
        # Flatten for processing
        batch_size = tf.shape(x)[0]
        features = tf.reshape(x, [batch_size, -1])
        
        # Build adjacency matrix
        adj = self.build_adjacency(batch_size)
        
        # Apply graph convolution
        gcn_output = self.gcn([features, adj])
        
        # Generate SRP parameters for each direction
        d_params = self.d_output(gcn_output)  # D0, Dc, D2c, D4c
        y_params = self.y_output(gcn_output)  # Y0, Ys
        b_params = self.b_output(gcn_output)  # B0, Bs
        
        return d_params, y_params, b_params
    
    def compute_accelerations(self, d_params, y_params, b_params, delta_u):
        """
        Compute accelerations using simplified ECOMC model (equation 4)
        
        Args:
            d_params: D direction parameters [D0, Dc, D2c, D4c]
            y_params: Y direction parameters [Y0, Ys]
            b_params: B direction parameters [B0, Bs]
            delta_u: Satellite relative orbit angle
            
        Returns:
            Accelerations in D, Y, B directions
        """
        # Extract parameters
        D0, Dc, D2c, D4c = tf.unstack(d_params, axis=-1)
        Y0, Ys = tf.unstack(y_params, axis=-1)
        B0, Bs = tf.unstack(b_params, axis=-1)
        
        # Convert to radians
        delta_u_rad = delta_u * np.pi / 180.0
        
        # Compute accelerations using simplified ECOMC model (equation 4)
        a_D = D0 + Dc * tf.cos(delta_u_rad) + D2c * tf.cos(2*delta_u_rad) + D4c * tf.cos(4*delta_u_rad)
        a_Y = Y0 + Ys * tf.sin(delta_u_rad)
        a_B = B0 + Bs * tf.sin(delta_u_rad)
        
        return a_D, a_Y, a_B
