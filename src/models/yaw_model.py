"""
LSTM-Attention model for yaw attitude correction
"""

import tensorflow as tf
from src.models.layers import EclipseAttention, DynamicsConstraint

class YawAttitudeModel(tf.keras.Model):
    """LSTM-Attention model for yaw attitude correction during eclipse periods"""
    
    def __init__(self, lstm_units=128, attention_units=64):
        """
        Initialize the YawAttitudeModel
        
        Args:
            lstm_units: Number of LSTM units
            attention_units: Number of attention units
        """
        super(YawAttitudeModel, self).__init__()
        
        # Input feature processing
        self.input_dense = tf.keras.layers.Dense(64, activation='relu')
        
        # LSTM layers for sequential modeling
        self.lstm1 = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(lstm_units // 2, return_sequences=True)
        
        # Custom attention mechanism focusing on eclipse periods
        self.attention = EclipseAttention(attention_units)
        
        # Output layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)  # Yaw angle correction
        
        # Physics constraint layer
        self.dynamics_constraint = DynamicsConstraint()
    
    def call(self, inputs, training=None):
        """
        Forward pass for the yaw attitude model
        
        Args:
            inputs: Input tensor with shape [batch, time_steps, features]
                   features = [time, beta, mu, nominal_yaw]
            
        Returns:
            Predicted yaw angles and physics constraint
        """
        # Extract individual features
        time_feature = inputs[:, :, 0:1]  # time
        beta = inputs[:, :, 1:2]  # sun elevation angle
        mu = inputs[:, :, 2:3]  # orbit angle
        nominal_yaw = inputs[:, :, 3:4]  # nominal yaw angle
        
        # Process input features
        x = self.input_dense(inputs)
        
        # Apply LSTM layers
        lstm_out1 = self.lstm1(x)
        lstm_out2 = self.lstm2(lstm_out1)
        
        # Apply attention focusing on eclipse periods
        context = self.attention(lstm_out2, beta=beta[:, :, 0])
        
        # Dense layers
        x = self.dense1(context)
        x = self.dense2(x)
        
        # Residual connection with nominal yaw
        # Extract the last timestep's nominal yaw for each sequence
        last_nominal_yaw = nominal_yaw[:, -1, :]
        
        # Output layer produces a correction to the nominal yaw
        yaw_correction = self.output_layer(x)
        
        # Add correction to nominal yaw
        predicted_yaw = last_nominal_yaw + yaw_correction
        
        # Apply physics constraint
        theoretical_yaw_rate, eclipse_mask = self.dynamics_constraint(
            predicted_yaw, 
            beta[:, -1, 0],  # Last timestep's beta 
            mu[:, -1, 0]     # Last timestep's mu
        )
        
        return predicted_yaw, theoretical_yaw_rate, eclipse_mask
