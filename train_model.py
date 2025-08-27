#!/usr/bin/env python3
"""
Train the BDS-3 MEO satellite yaw attitude and SRP model
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
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train BDS-3 MEO satellite yaw attitude and SRP model'
    )
    parser.add_argument('--data-dir', default='obx-dataset',
                       help='Directory containing OBX data')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--target-sats', default='C23,C25',
                       help='Comma-separated list of target satellites')
    parser.add_argument('--checkpoint', default=None,
                       help='Path to checkpoint file to resume from')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Parse target satellites
    target_satellites = args.target_sats.split(',')
    
    # Process OBX data
    print("Processing OBX data...")
    processor = OBXProcessor(args.data_dir)
    processed_data = processor.process_dataset(target_sats=target_satellites)
    
    # Prepare ML dataset
    print("Preparing ML dataset...")
    features, targets = processor.prepare_ml_dataset(processed_data)
    
    # Create data generators
    print("Creating data generators...")
    (train_gen_yaw, val_gen_yaw), (train_gen_srp, val_gen_srp) = processor.get_data_generators(
        features, targets, batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = CombinedModel(
        yaw_lstm_units=128,
        yaw_attention_units=64,
        srp_gcn_filters=64,
        srp_gcn_kernel=3
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        log_dir=os.path.join(args.output_dir, 'logs')
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        trainer.load_weights(args.checkpoint)
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_data=(train_gen_yaw, train_gen_srp),
        val_data=(val_gen_yaw, val_gen_srp),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, 'final_model')
    trainer.save_model(model_path)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(
        model=model,
        processed_data=processed_data,
        output_dir=args.output_dir
    )
    
    # Generate metrics for each satellite
    for center in ['WHU_FIN', 'WHU_RAP']:
        for sat in target_satellites:
            if sat in processed_data[center]:
                print(f"Evaluating {center} {sat}...")
                metrics = evaluator.evaluate_metrics(center=center, satellite=sat)
                
                print(f"Metrics for {center} {sat}:")
                print(f"  Yaw RMSE: {metrics['yaw_rmse_deg']:.4f} deg")
                print(f"  Radial Error: {metrics['radial_error_cm']:.4f} cm (Improvement: {metrics['radial_improvement']:.2f}%)")
                print(f"  Normal Error: {metrics['normal_error_cm']:.4f} cm (Improvement: {metrics['normal_improvement']:.2f}%)")
                print(f"  SLR Residual: {metrics['slr_residual_cm']:.4f} cm (Improvement: {metrics['slr_improvement']:.2f}%)")
                print(f"  Day Boundary Jump: {metrics['day_boundary_jump_cm']:.4f} cm (Improvement: {metrics['jump_improvement']:.2f}%)")
                
                # Generate visualizations
                print(f"Generating visualizations for {center} {sat}...")
                evaluator.visualize_yaw_comparison(center=center, satellite=sat)
    
    # Generate SRP parameter visualization
    evaluator.visualize_srp_parameters(center='WHU_FIN', satellites=target_satellites)
    
    print(f"Training and evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
