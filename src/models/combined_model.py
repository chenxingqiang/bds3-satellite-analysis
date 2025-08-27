"""
Combined model integrating yaw attitude and SRP compensation branches
"""

import tensorflow as tf
from src.models.yaw_model import YawAttitudeModel
from src.models.srp_model import SRPCompensationModel

class CombinedModel(tf.keras.Model):
    """
    Combined model integrating yaw attitude and SRP compensation branches
    This model follows the paper's approach of combining attitude correction and SRP modeling
    """
    
    def __init__(self, yaw_lstm_units=128, yaw_attention_units=64,
                srp_gcn_filters=64, srp_gcn_kernel=3):
        """
        Initialize the combined model
        
        Args:
            yaw_lstm_units: Number of LSTM units in yaw model
            yaw_attention_units: Number of attention units in yaw model
            srp_gcn_filters: Number of GCN filters in SRP model
            srp_gcn_kernel: Kernel size for GCN in SRP model
        """
        super(CombinedModel, self).__init__()
        
        # Initialize individual models
        self.yaw_model = YawAttitudeModel(
            lstm_units=yaw_lstm_units,
            attention_units=yaw_attention_units
        )
        
        self.srp_model = SRPCompensationModel(
            gcn_filters=srp_gcn_filters,
            gcn_kernel=srp_gcn_kernel
        )
    
    def call(self, inputs):
        """
        Forward pass for the combined model
        
        Args:
            inputs: Dictionary with keys 'yaw' and 'srp' containing inputs for each branch
            
        Returns:
            Dictionary with outputs from both branches
        """
        # Process yaw branch
        yaw_pred, theoretical_yaw_rate, eclipse_mask = self.yaw_model(inputs['yaw'])
        
        # Process SRP branch
        d_params, y_params, b_params = self.srp_model(inputs['srp'])
        
        # Extract orbit angle from SRP inputs for acceleration computation
        delta_u = inputs['srp'][:, -1, 2]  # Last timestep, orbit angle feature
        
        # Compute accelerations using simplified ECOMC model
        a_D, a_Y, a_B = self.srp_model.compute_accelerations(d_params, y_params, b_params, delta_u)
        
        return {
            'yaw': {
                'prediction': yaw_pred,
                'theoretical_rate': theoretical_yaw_rate,
                'eclipse_mask': eclipse_mask
            },
            'srp': {
                'd_params': d_params,
                'y_params': y_params,
                'b_params': b_params,
                'accelerations': [a_D, a_Y, a_B]
            }
        }
    
    def train_step(self, data):
        """
        Custom training step with physics-guided loss
        
        Args:
            data: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of losses
        """
        inputs, targets = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(inputs, training=True)
            
            # Yaw branch loss
            yaw_pred = outputs['yaw']['prediction']
            yaw_true = targets['yaw']
            yaw_loss = tf.reduce_mean(tf.square(yaw_pred - yaw_true))
            
            # Physics constraint loss for yaw
            theoretical_rate = outputs['yaw']['theoretical_rate']
            eclipse_mask = outputs['yaw']['eclipse_mask']
            
            # Only apply dynamics constraint outside eclipse (when eclipse_mask == 0)
            dynamics_loss = tf.reduce_mean(
                (1 - eclipse_mask) * tf.square(theoretical_rate)
            )
            
            # SRP branch loss - use physics-based constraints
            # Instead of direct supervision (since we don't have ground truth SRP values)
            srp_params_regularization = tf.reduce_mean(
                tf.square(outputs['srp']['d_params']) +
                tf.square(outputs['srp']['y_params']) +
                tf.square(outputs['srp']['b_params'])
            ) * 0.01  # Regularization weight
            
            # Total loss
            loss = yaw_loss + 0.1 * dynamics_loss + srp_params_regularization
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(yaw_true, yaw_pred)
        
        # Return losses
        return {
            'loss': loss,
            'yaw_loss': yaw_loss,
            'dynamics_loss': dynamics_loss,
            'srp_regularization': srp_params_regularization
        }
    
    def test_step(self, data):
        """
        Custom test step
        
        Args:
            data: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of metrics
        """
        inputs, targets = data
        
        # Forward pass
        outputs = self(inputs, training=False)
        
        # Yaw branch metrics
        yaw_pred = outputs['yaw']['prediction']
        yaw_true = targets['yaw']
        yaw_mse = tf.reduce_mean(tf.square(yaw_pred - yaw_true))
        
        # Update metrics
        self.compiled_metrics.update_state(yaw_true, yaw_pred)
        
        # Return metrics
        return {
            'yaw_mse': yaw_mse
        }
    
    def get_config(self):
        """Get config for serialization"""
        config = super(CombinedModel, self).get_config()
        config.update({
            'yaw_lstm_units': self.yaw_model.lstm1.units,
            'yaw_attention_units': self.yaw_model.attention.attention_dense.units,
            'srp_gcn_filters': self.srp_model.gcn.channels,
            'srp_gcn_kernel': self.srp_model.gcn.kernel_size
        })
        return config
