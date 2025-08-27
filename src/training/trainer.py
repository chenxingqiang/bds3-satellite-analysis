"""
Model trainer for BDS-3 MEO satellite yaw attitude and SRP model
"""

import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from tqdm import tqdm

from src.models.combined_model import CombinedModel
from src.training.losses import PhysicsGuidedLoss, CompositeLoss

class ModelTrainer:
    """Trainer for the combined satellite model"""
    
    def __init__(self, model: CombinedModel, log_dir: str = 'logs'):
        """
        Initialize the trainer
        
        Args:
            model: Combined model to train
            log_dir: Directory for storing logs
        """
        self.model = model
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup tensorboard logging
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
        
        # Setup model checkpoint
        self.checkpoint_path = os.path.join(self.log_dir, 'model_checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_path, 'model_{epoch:02d}'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    
    def train(self, 
              train_data: Tuple, 
              val_data: Tuple, 
              epochs: int = 50, 
              batch_size: int = 32,
              learning_rate: float = 1e-3):
        """
        Train the model
        
        Args:
            train_data: Tuple of (train_gen_yaw, train_gen_srp)
            val_data: Tuple of (val_gen_yaw, val_gen_srp)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            
        Returns:
            Training history
        """
        # Unpack data generators
        train_gen_yaw, train_gen_srp = train_data
        val_gen_yaw, val_gen_srp = val_data
        
        # Setup optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=len(train_gen_yaw) * 5,  # Decay every 5 epochs
            decay_rate=0.9,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile model with custom losses
        self.model.compile(
            optimizer=optimizer,
            loss={
                'yaw': PhysicsGuidedLoss(yaw_weight=0.6, srp_weight=0.4),
                'srp': CompositeLoss(physics_weight=0.3, data_weight=0.7)
            },
            metrics={
                'yaw': ['mae', 'mse']
            }
        )
        
        # Learning rate callback
        lr_callback = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.9 if epoch > 0 and epoch % 10 == 0 else lr
        )
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Custom train loop to handle multiple inputs
        history = {'loss': [], 'val_loss': [], 'yaw_loss': [], 'val_yaw_loss': []}
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training loop
            train_losses = []
            train_yaw_losses = []
            
            for batch in tqdm(range(len(train_gen_yaw)), desc="Training"):
                # Get batch data
                yaw_features, yaw_targets = train_gen_yaw[batch]
                srp_features, _ = train_gen_srp[batch]  # We don't have direct SRP targets
                
                # Prepare inputs and targets
                inputs = {
                    'yaw': yaw_features,
                    'srp': srp_features
                }
                
                targets = {
                    'yaw': yaw_targets
                }
                
                # Train step
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs, training=True)
                    
                    # Compute losses
                    yaw_loss = tf.reduce_mean(
                        tf.square(outputs['yaw']['prediction'] - targets['yaw'])
                    )
                    
                    # Physics constraints
                    dynamics_loss = tf.reduce_mean(
                        (1 - outputs['yaw']['eclipse_mask']) * 
                        tf.square(outputs['yaw']['theoretical_rate'])
                    )
                    
                    srp_reg_loss = tf.reduce_mean(
                        tf.square(outputs['srp']['d_params']) +
                        tf.square(outputs['srp']['y_params']) +
                        tf.square(outputs['srp']['b_params'])
                    ) * 0.01
                    
                    # Total loss
                    total_loss = yaw_loss + 0.1 * dynamics_loss + srp_reg_loss
                
                # Compute gradients and update weights
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                train_losses.append(float(total_loss))
                train_yaw_losses.append(float(yaw_loss))
            
            # Validation loop
            val_losses = []
            val_yaw_losses = []
            
            for batch in tqdm(range(len(val_gen_yaw)), desc="Validation"):
                # Get batch data
                yaw_features, yaw_targets = val_gen_yaw[batch]
                srp_features, _ = val_gen_srp[batch]
                
                # Prepare inputs and targets
                inputs = {
                    'yaw': yaw_features,
                    'srp': srp_features
                }
                
                targets = {
                    'yaw': yaw_targets
                }
                
                # Validation step
                outputs = self.model(inputs, training=False)
                
                # Compute losses
                yaw_loss = tf.reduce_mean(
                    tf.square(outputs['yaw']['prediction'] - targets['yaw'])
                )
                
                # Physics constraints
                dynamics_loss = tf.reduce_mean(
                    (1 - outputs['yaw']['eclipse_mask']) * 
                    tf.square(outputs['yaw']['theoretical_rate'])
                )
                
                srp_reg_loss = tf.reduce_mean(
                    tf.square(outputs['srp']['d_params']) +
                    tf.square(outputs['srp']['y_params']) +
                    tf.square(outputs['srp']['b_params'])
                ) * 0.01
                
                # Total loss
                total_loss = yaw_loss + 0.1 * dynamics_loss + srp_reg_loss
                
                val_losses.append(float(total_loss))
                val_yaw_losses.append(float(yaw_loss))
            
            # Update history
            history['loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            history['yaw_loss'].append(np.mean(train_yaw_losses))
            history['val_yaw_loss'].append(np.mean(val_yaw_losses))
            
            # Print metrics
            print(f"Train Loss: {history['loss'][-1]:.4f}, "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, "
                  f"Train Yaw Loss: {history['yaw_loss'][-1]:.4f}, "
                  f"Val Yaw Loss: {history['val_yaw_loss'][-1]:.4f}")
            
            # Save model checkpoint
            self.model.save_weights(
                os.path.join(self.checkpoint_path, f'model_epoch_{epoch+1:02d}.h5')
            )
            
            # Check for early stopping
            if epoch > 10 and history['val_loss'][-1] > history['val_loss'][-2] > history['val_loss'][-3]:
                print("Early stopping triggered")
                break
        
        return history
    
    def save_model(self, path: str):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_weights(self, checkpoint_path: str):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        self.model.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
