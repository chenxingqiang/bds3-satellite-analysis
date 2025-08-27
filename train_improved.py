#!/usr/bin/env python3
"""
Improved training script for BDS-3 MEO satellite yaw attitude analysis
Based on error analysis, this version addresses the performance issues
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

from src.data.obx_processor import OBXProcessor
from src.models.combined_model import CombinedModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

def main():
    # Parse arguments with improved defaults
    parser = argparse.ArgumentParser(
        description='Improved training for BDS-3 MEO satellite model'
    )
    parser.add_argument('--data-dir', default='obx-dataset',
                       help='Directory containing OBX data')
    parser.add_argument('--output-dir', default='results_improved',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50,  # Increased from 1-3
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,  # Smaller batch for stability
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001,  # Reduced from 0.001
                       help='Initial learning rate')
    parser.add_argument('--target-sats', default='C23,C25',
                       help='Comma-separated list of target satellites')
    parser.add_argument('--physics-weight', type=float, default=1.0,  # Increased physics weight
                       help='Weight for physics constraint loss')
    args = parser.parse_args()
    
    print("🚀 启动改进版 BDS-3 MEO 卫星训练")
    print("=" * 50)
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批次大小: {args.batch_size}")
    print(f"物理约束权重: {args.physics_weight}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Parse target satellites
    target_satellites = args.target_sats.split(',')
    
    # Process OBX data
    print("\n📊 处理 OBX 数据...")
    processor = OBXProcessor(args.data_dir)
    processed_data = processor.process_dataset(target_sats=target_satellites)
    
    # Prepare ML dataset
    print("🔧 准备机器学习数据集...")
    features, targets = processor.prepare_ml_dataset(processed_data)
    
    # Create data generators with validation split
    print("📦 创建数据生成器...")
    (train_gen_yaw, val_gen_yaw), (train_gen_srp, val_gen_srp) = processor.get_data_generators(
        features, targets, batch_size=args.batch_size, val_split=0.2
    )
    
    print(f"训练样本数: {len(train_gen_yaw) * args.batch_size}")
    print(f"验证样本数: {len(val_gen_yaw) * args.batch_size}")
    
    # Create improved model with regularization
    print("🏗️ 创建改进模型...")
    model = CombinedModel(
        yaw_lstm_units=64,  # Reduced to prevent overfitting
        yaw_attention_units=32,
        srp_gcn_filters=32,
        srp_gcn_kernel=3
    )
    
    # Create custom trainer with improved settings
    class ImprovedTrainer(ModelTrainer):
        def create_optimizers(self, learning_rate):
            # Use cosine decay learning rate schedule
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=len(train_gen_yaw) * args.epochs
            )
            return tf.keras.optimizers.AdamW(
                learning_rate=cosine_decay,
                weight_decay=0.01  # L2 regularization
            )
        
        def improved_train_step(self, inputs, targets, physics_weight):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                
                # Enhanced loss computation
                yaw_pred = outputs['yaw']['prediction']
                yaw_true = targets['yaw']
                
                # Huber loss for robustness to outliers
                yaw_loss = tf.reduce_mean(tf.keras.losses.huber(yaw_true, yaw_pred, delta=1.0))
                
                # Enhanced physics constraints
                theoretical_rate = outputs['yaw']['theoretical_rate']
                eclipse_mask = outputs['yaw']['eclipse_mask']
                
                # Physics loss with higher weight
                dynamics_loss = tf.reduce_mean(
                    (1 - eclipse_mask) * tf.square(theoretical_rate)
                ) * physics_weight
                
                # L2 regularization on model weights
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables 
                                   if 'bias' not in v.name]) * 0.001
                
                # SRP regularization
                d_reg = tf.reduce_mean(tf.square(outputs['srp']['d_params']))
                y_reg = tf.reduce_mean(tf.square(outputs['srp']['y_params']))
                b_reg = tf.reduce_mean(tf.square(outputs['srp']['b_params']))
                srp_reg_loss = (d_reg + y_reg + b_reg) * 0.01
                
                # Total loss
                total_loss = yaw_loss + dynamics_loss + l2_loss + srp_reg_loss
            
            # Compute gradients with clipping
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]  # Gradient clipping
            
            return total_loss, yaw_loss, dynamics_loss, gradients
    
    # Create improved trainer
    trainer = ImprovedTrainer(
        model=model,
        log_dir=os.path.join(args.output_dir, 'logs')
    )
    
    # Custom training loop with improvements
    optimizer = trainer.create_optimizers(args.learning_rate)
    
    print(f"\n🎯 开始改进训练 ({args.epochs} 轮)...")
    history = {'loss': [], 'val_loss': [], 'yaw_loss': [], 'val_yaw_loss': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_losses = []
        train_yaw_losses = []
        
        for batch_idx in range(len(train_gen_yaw)):
            # Get batch data
            yaw_features, yaw_targets = train_gen_yaw[batch_idx]
            srp_features, _ = train_gen_srp[batch_idx]
            
            inputs = {'yaw': yaw_features, 'srp': srp_features}
            targets = {'yaw': yaw_targets}
            
            # Improved training step
            total_loss, yaw_loss, dynamics_loss, gradients = trainer.improved_train_step(
                inputs, targets, args.physics_weight
            )
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_losses.append(float(total_loss))
            train_yaw_losses.append(float(yaw_loss))
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_gen_yaw)}, "
                      f"Loss: {total_loss:.4f}, Yaw: {yaw_loss:.4f}, Physics: {dynamics_loss:.4f}")
        
        # Validation
        val_losses = []
        val_yaw_losses = []
        
        for batch_idx in range(len(val_gen_yaw)):
            yaw_features, yaw_targets = val_gen_yaw[batch_idx]
            srp_features, _ = val_gen_srp[batch_idx]
            
            inputs = {'yaw': yaw_features, 'srp': srp_features}
            outputs = model(inputs, training=False)
            
            # Validation loss
            yaw_pred = outputs['yaw']['prediction']
            yaw_true = yaw_targets
            val_yaw_loss = tf.reduce_mean(tf.keras.losses.huber(yaw_true, yaw_pred, delta=1.0))
            
            theoretical_rate = outputs['yaw']['theoretical_rate']
            eclipse_mask = outputs['yaw']['eclipse_mask']
            val_dynamics_loss = tf.reduce_mean(
                (1 - eclipse_mask) * tf.square(theoretical_rate)
            ) * args.physics_weight
            
            val_total_loss = val_yaw_loss + val_dynamics_loss
            
            val_losses.append(float(val_total_loss))
            val_yaw_losses.append(float(val_yaw_loss))
        
        # Update history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_yaw = np.mean(train_yaw_losses)
        epoch_val_yaw = np.mean(val_yaw_losses)
        
        history['loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['yaw_loss'].append(epoch_train_yaw)
        history['val_yaw_loss'].append(epoch_val_yaw)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print(f"Train Yaw: {epoch_train_yaw:.4f}, Val Yaw: {epoch_val_yaw:.4f}")
        
        # Early stopping and best model saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            model.save_weights(os.path.join(args.output_dir, 'best_model.weights.h5'))
            print("💾 保存最佳模型")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and save
    model.load_weights(os.path.join(args.output_dir, 'best_model.weights.h5'))
    model_path = os.path.join(args.output_dir, 'final_model.keras')
    model.save(model_path)
    
    print(f"\n✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存至: {model_path}")
    
    # Simple evaluation (without problematic SRP visualization)
    print("\n🔍 评估模型...")
    evaluator = ModelEvaluator(
        model=model,
        processed_data=processed_data,
        output_dir=args.output_dir
    )
    
    # Only evaluate metrics, skip problematic visualizations
    for center in ['WHU_FIN', 'WHU_RAP']:
        for sat in target_satellites:
            if sat in processed_data[center]:
                print(f"\n评估 {center} {sat}...")
                metrics = evaluator.evaluate_metrics(center=center, satellite=sat)
                
                print(f"  偏航角RMSE: {metrics['yaw_rmse_deg']:.4f}°")
                print(f"  径向误差: {metrics['radial_error_cm']:.2f} cm")
                print(f"  法向误差: {metrics['normal_error_cm']:.2f} cm") 
                print(f"  SLR残差: {metrics['slr_residual_cm']:.2f} cm")
                
                # Generate yaw comparison plots only
                try:
                    evaluator.visualize_yaw_comparison(center=center, satellite=sat)
                    print(f"  ✅ 生成了 {center}_{sat} 偏航角对比图")
                except Exception as e:
                    print(f"  ⚠️ 可视化失败: {e}")
    
    print(f"\n🎉 改进训练完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
