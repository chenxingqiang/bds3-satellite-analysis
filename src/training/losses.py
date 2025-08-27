"""
Custom loss functions for physics-guided training
"""

import tensorflow as tf
import numpy as np

class PhysicsGuidedLoss(tf.keras.losses.Loss):
    """
    Physics-guided loss function combining data-driven and physics-based constraints
    """
    
    def __init__(self, yaw_weight=0.6, srp_weight=0.4):
        """
        Initialize the physics-guided loss
        
        Args:
            yaw_weight: Weight for yaw branch loss
            srp_weight: Weight for SRP branch loss
        """
        super(PhysicsGuidedLoss, self).__init__()
        self.yaw_weight = yaw_weight
        self.srp_weight = srp_weight
    
    def call(self, y_true, y_pred):
        """
        Compute the physics-guided loss
        
        Args:
            y_true: Dictionary containing ground truth values
            y_pred: Dictionary containing predicted values
            
        Returns:
            Combined loss value
        """
        # Yaw branch loss
        yaw_true = y_true['yaw']
        yaw_pred = y_pred['yaw']['prediction']
        yaw_loss = tf.reduce_mean(tf.square(yaw_pred - yaw_true))
        
        # Physics constraint losses
        theoretical_rate = y_pred['yaw']['theoretical_rate']
        eclipse_mask = y_pred['yaw']['eclipse_mask']
        
        # Only apply dynamics constraint outside eclipse
        dynamics_loss = tf.reduce_mean(
            (1 - eclipse_mask) * tf.square(theoretical_rate)
        )
        
        # SRP model regularization
        srp_params_regularization = tf.reduce_mean(
            tf.square(y_pred['srp']['d_params']) +
            tf.square(y_pred['srp']['y_params']) +
            tf.square(y_pred['srp']['b_params'])
        ) * 0.01  # Regularization weight
        
        # Combined loss
        combined_loss = (
            self.yaw_weight * (yaw_loss + 0.1 * dynamics_loss) +
            self.srp_weight * srp_params_regularization
        )
        
        return combined_loss

class RadiationPressureConstraint(tf.keras.losses.Loss):
    """
    Radiation pressure physics constraint based on ECOMC model structure
    """
    
    def __init__(self):
        """Initialize the radiation pressure constraint loss"""
        super(RadiationPressureConstraint, self).__init__()
    
    def call(self, y_true, y_pred):
        """
        Compute the radiation pressure constraint loss
        
        Args:
            y_true: Ground truth (not used, included for API compatibility)
            y_pred: Dictionary containing predicted SRP parameters
            
        Returns:
            Physics constraint loss value
        """
        # Extract SRP parameters
        d_params = y_pred['srp']['d_params']
        y_params = y_pred['srp']['y_params']
        b_params = y_pred['srp']['b_params']
        
        # Physics-based constraints from equations (5-8)
        # 1. Relationship between D and B components (simplified)
        d0 = d_params[:, 0]  # D0
        b0 = b_params[:, 0]  # B0
        
        # In eclipse period, D0 and B0 should have a specific relationship
        # based on the cube model in equations (5-6)
        db_ratio_loss = tf.reduce_mean(tf.square(d0 - 3*b0))
        
        # 2. Periodicity constraint (Dc, D2c, D4c should show decreasing amplitude)
        dc = tf.abs(d_params[:, 1])   # Dc
        d2c = tf.abs(d_params[:, 2])  # D2c
        d4c = tf.abs(d_params[:, 3])  # D4c
        
        periodicity_loss = tf.reduce_mean(
            tf.maximum(0.0, d2c - dc) + tf.maximum(0.0, d4c - d2c)
        )
        
        # 3. Symmetry constraint for Y parameters
        y0 = y_params[:, 0]  # Y0
        ys = y_params[:, 1]  # Ys
        
        # Y0 should be small due to satellite symmetry
        symmetry_loss = tf.reduce_mean(tf.square(y0))
        
        # Combined physics constraint loss
        physics_loss = db_ratio_loss + periodicity_loss + symmetry_loss
        
        return physics_loss

class CompositeLoss(tf.keras.losses.Loss):
    """
    Composite loss combining data fitting and physics constraints
    """
    
    def __init__(self, physics_weight=0.3, data_weight=0.7):
        """
        Initialize the composite loss
        
        Args:
            physics_weight: Weight for physics constraint loss
            data_weight: Weight for data fitting loss
        """
        super(CompositeLoss, self).__init__()
        self.physics_weight = physics_weight
        self.data_weight = data_weight
        self.physics_loss = RadiationPressureConstraint()
        self.data_loss = tf.keras.losses.Huber()
    
    def call(self, y_true, y_pred):
        """
        Compute the composite loss
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Combined loss value
        """
        # Physics constraint component
        physics_component = self.physics_loss(y_true, y_pred)
        
        # Data fitting component (if available)
        # For SRP we might not have direct supervision, so we use a proxy
        if isinstance(y_true, dict) and 'srp' in y_true:
            data_component = self.data_loss(y_true['srp'], y_pred['srp']['accelerations'])
        else:
            # If no direct supervision, use a regularization term
            data_component = tf.reduce_mean(
                tf.square(y_pred['srp']['d_params']) + 
                tf.square(y_pred['srp']['y_params']) + 
                tf.square(y_pred['srp']['b_params'])
            )
        
        # Combined loss
        return self.physics_weight * physics_component + self.data_weight * data_component
