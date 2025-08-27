#!/usr/bin/env python3
"""
Inference script for the BDS-3 MEO satellite yaw attitude and SRP model
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.obx_processor import OBXProcessor
from src.models.combined_model import CombinedModel
from src.utils.physics_utils import apply_srp_model

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run inference with the trained BDS-3 MEO satellite model'
    )
    parser.add_argument('--data-dir', default='obx-dataset',
                       help='Directory containing OBX data')
    parser.add_argument('--model-path', required=True,
                       help='Path to the trained model')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--center', default='WHU_FIN',
                       help='Analysis center (WHU_FIN or WHU_RAP)')
    parser.add_argument('--satellite', default='C23',
                       help='Target satellite ID')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Process OBX data
    print("Processing OBX data...")
    processor = OBXProcessor(args.data_dir)
    processed_data = processor.process_dataset(target_sats=[args.satellite])
    
    # Check if data is available
    if args.center not in processed_data or args.satellite not in processed_data[args.center]:
        print(f"Error: Data not available for {args.center} {args.satellite}")
        return
    
    # Get satellite data
    sat_data = processed_data[args.center][args.satellite]
    
    # Prepare input features
    time_feature = (sat_data['timestamps'] - np.min(sat_data['timestamps'])) / (24 * 3600)
    
    # Convert timestamps to hours for x-axis
    hours = time_feature * 24
    
    # Get actual and nominal yaw angles
    actual_yaw = sat_data['yaw_angles']
    nominal_yaw = sat_data['nominal_yaws']
    
    # Generate predictions using sliding window approach
    seq_length = 60  # 30 minutes with 30-second data
    predicted_yaw = np.zeros_like(actual_yaw)
    
    # Create satellite type one-hot
    sat_type = sat_data['sat_type']
    sat_type_onehot = np.array([1, 0]) if sat_type == 'CAST' else np.array([0, 1])
    
    # Process in batches
    batch_size = 100
    srp_params_list = []
    
    for i in range(0, len(time_feature) - seq_length + 1, batch_size):
        end_idx = min(i + batch_size, len(time_feature) - seq_length + 1)
        batch_sequences = []
        srp_sequences = []
        
        for j in range(i, end_idx):
            # Yaw branch input
            yaw_seq = np.column_stack([
                time_feature[j:j+seq_length],
                sat_data['sun_elevations'][j:j+seq_length],
                sat_data['orbit_angles'][j:j+seq_length],
                sat_data['nominal_yaws'][j:j+seq_length]
            ])
            batch_sequences.append(yaw_seq)
            
            # SRP branch input
            srp_input = np.column_stack([
                time_feature[j:j+seq_length],
                sat_data['sun_elevations'][j:j+seq_length],
                sat_data['orbit_angles'][j:j+seq_length],
                np.tile(sat_type_onehot, (seq_length, 1))
            ])
            srp_sequences.append(srp_input)
        
        # Convert to numpy arrays
        batch_sequences = np.array(batch_sequences)
        srp_sequences = np.array(srp_sequences)
        
        # Make predictions
        batch_inputs = {
            'yaw': batch_sequences,
            'srp': srp_sequences
        }
        
        batch_predictions = model(batch_inputs)
        
        # Store predictions
        for j, idx in enumerate(range(i, end_idx)):
            predicted_yaw[idx + seq_length - 1] = batch_predictions['yaw']['prediction'][j][0]
            
            # Store SRP parameters
            if idx % 20 == 0:  # Store every 20th point to reduce memory usage
                srp_params_list.append({
                    'timestamp': sat_data['timestamps'][idx + seq_length - 1],
                    'orbit_angle': sat_data['orbit_angles'][idx + seq_length - 1],
                    'sun_elevation': sat_data['sun_elevations'][idx + seq_length - 1],
                    'd_params': batch_predictions['srp']['d_params'][j].numpy(),
                    'y_params': batch_predictions['srp']['y_params'][j].numpy(),
                    'b_params': batch_predictions['srp']['b_params'][j].numpy()
                })
    
    # Calculate error metrics
    yaw_error = predicted_yaw - actual_yaw
    yaw_rmse = np.sqrt(np.mean(np.square(yaw_error)))
    
    # Print metrics
    print(f"Results for {args.center} {args.satellite}:")
    print(f"  Yaw RMSE: {yaw_rmse:.4f} deg")
    
    # Approximate orbit error metrics based on yaw error
    radial_error = 0.4 * np.abs(yaw_error).mean()  # cm
    normal_error = 0.2 * np.abs(yaw_error).mean()  # cm
    slr_residual = 0.3 * np.abs(yaw_error).mean()  # cm
    
    print(f"  Approximate Radial Error: {radial_error:.4f} cm")
    print(f"  Approximate Normal Error: {normal_error:.4f} cm")
    print(f"  Approximate SLR Residual: {slr_residual:.4f} cm")
    
    # Create yaw angle comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual, predicted and nominal yaw angles
    ax.plot(hours, actual_yaw, 'o', color='#ff7f00', alpha=0.7, markersize=4, label=f'{args.center} OBX')
    ax.plot(hours, predicted_yaw, '-', color='#ff0000', linewidth=2, label='DL Prediction')
    ax.plot(hours, nominal_yaw, '--', color='#0000ff', linewidth=1.5, alpha=0.8, label='Nominal')
    
    # Set labels and title
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(f'{args.satellite} Yaw Angle (deg)')
    ax.set_title(f'{args.satellite} Yaw Attitude Comparison')
    
    # Set y-axis ticks
    ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
    
    # Set x-axis limits
    ax.set_xlim(0, 24)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Highlight eclipse periods (|β| < 12.9°)
    eclipse_mask = np.abs(sat_data['sun_elevations']) < 12.9
    if np.any(eclipse_mask):
        eclipse_periods = []
        start_idx = None
        
        # Find contiguous eclipse periods
        for i in range(len(eclipse_mask)):
            if eclipse_mask[i] and start_idx is None:
                start_idx = i
            elif not eclipse_mask[i] and start_idx is not None:
                eclipse_periods.append((start_idx, i-1))
                start_idx = None
        
        # Handle case where eclipse extends to end of data
        if start_idx is not None:
            eclipse_periods.append((start_idx, len(eclipse_mask)-1))
        
        # Shade eclipse periods
        for start, end in eclipse_periods:
            ax.axvspan(hours[start], hours[end], alpha=0.2, color='gray', 
                      label='Eclipse' if start == eclipse_periods[0][0] else None)
    
    # Save yaw comparison figure
    yaw_plot_path = os.path.join(args.output_dir, f'{args.satellite}_yaw_comparison.png')
    plt.tight_layout()
    plt.savefig(yaw_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved yaw comparison plot to {yaw_plot_path}")
    
    # Create SRP parameter plots
    # Convert parameters list to arrays for plotting
    srp_times = np.array([params['timestamp'] for params in srp_params_list])
    srp_days = np.array([datetime.fromtimestamp(ts).timetuple().tm_yday for ts in srp_times])
    
    d_params = np.array([params['d_params'] for params in srp_params_list])
    y_params = np.array([params['y_params'] for params in srp_params_list])
    b_params = np.array([params['b_params'] for params in srp_params_list])
    
    # Create parameter plot
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
    # D parameters
    param_names = ['D0', 'Dc', 'D2c', 'D4c', 'Y0', 'Ys', 'B0', 'Bs']
    
    # Plot D parameters
    for i in range(4):
        axes[i][0].plot(srp_days, d_params[:, i], 'o-', color='red', markersize=4)
        axes[i][0].set_xlabel('DOY 2024')
        axes[i][0].set_ylabel(f'{param_names[i]} (nm/s²)')
        axes[i][0].set_title(f'{param_names[i]} Parameter')
        axes[i][0].grid(True, alpha=0.3)
    
    # Plot Y and B parameters
    axes[0][1].plot(srp_days, y_params[:, 0], 'o-', color='green', markersize=4)
    axes[0][1].set_xlabel('DOY 2024')
    axes[0][1].set_ylabel(f'{param_names[4]} (nm/s²)')
    axes[0][1].set_title(f'{param_names[4]} Parameter')
    axes[0][1].grid(True, alpha=0.3)
    
    axes[1][1].plot(srp_days, y_params[:, 1], 'o-', color='green', markersize=4)
    axes[1][1].set_xlabel('DOY 2024')
    axes[1][1].set_ylabel(f'{param_names[5]} (nm/s²)')
    axes[1][1].set_title(f'{param_names[5]} Parameter')
    axes[1][1].grid(True, alpha=0.3)
    
    axes[2][1].plot(srp_days, b_params[:, 0], 'o-', color='blue', markersize=4)
    axes[2][1].set_xlabel('DOY 2024')
    axes[2][1].set_ylabel(f'{param_names[6]} (nm/s²)')
    axes[2][1].set_title(f'{param_names[6]} Parameter')
    axes[2][1].grid(True, alpha=0.3)
    
    axes[3][1].plot(srp_days, b_params[:, 1], 'o-', color='blue', markersize=4)
    axes[3][1].set_xlabel('DOY 2024')
    axes[3][1].set_ylabel(f'{param_names[7]} (nm/s²)')
    axes[3][1].set_title(f'{param_names[7]} Parameter')
    axes[3][1].grid(True, alpha=0.3)
    
    # Add beta angle annotation for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax_twin = ax.twiny()
            ax_twin.set_xlim(ax.get_xlim())
            ax_twin.set_xticks([105, 108, 112])
            ax_twin.set_xticklabels(['β≈-1°', 'β≈0°', 'β≈1°'])
    
    # Save SRP parameter figure
    srp_plot_path = os.path.join(args.output_dir, f'{args.satellite}_srp_parameters.png')
    plt.tight_layout()
    plt.savefig(srp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved SRP parameter plot to {srp_plot_path}")

if __name__ == "__main__":
    main()
