#!/usr/bin/env python3
"""
iTransformer PyTorch training script for BDS-3 MEO satellite yaw attitude analysis
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.itransformer_pytorch import iTransformerCombinedModel
from src.data.pytorch_dataset import SatelliteDataModule, collate_fn
from src.evaluation.evaluator import ModelEvaluator

class PhysicsConstrainedLoss(nn.Module):
    """Custom loss with physics constraints for satellite dynamics"""

    def __init__(self, physics_weight: float = 1.0, l2_weight: float = 0.001):
        super().__init__()
        self.physics_weight = physics_weight
        self.l2_weight = l2_weight
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)

    def forward(self, predictions, targets, model):
        # Yaw prediction loss (Huber for robustness)
        yaw_pred = predictions['yaw']['prediction']
        yaw_true = targets['yaw']
        yaw_loss = self.huber_loss(yaw_pred, yaw_true)

        # Physics constraint loss
        theoretical_rate = predictions['yaw']['theoretical_rate']
        eclipse_mask = predictions['yaw']['eclipse_mask']

        # During non-eclipse periods, theoretical rate should be small
        non_eclipse_mask = ~eclipse_mask
        physics_loss = torch.mean(
            non_eclipse_mask.float() * torch.square(theoretical_rate)
        ) * self.physics_weight

        # L2 regularization
        l2_reg = sum(torch.sum(torch.square(param)) for param in model.parameters())
        l2_loss = l2_reg * self.l2_weight

        # SRP regularization
        srp_outputs = predictions['srp']
        srp_reg = (
            torch.mean(torch.square(srp_outputs['d_params'])) +
            torch.mean(torch.square(srp_outputs['y_params'])) +
            torch.mean(torch.square(srp_outputs['b_params']))
        ) * 0.01

        # Total loss
        total_loss = yaw_loss + physics_loss + l2_loss + srp_reg

        return {
            'total': total_loss,
            'yaw': yaw_loss,
            'physics': physics_loss,
            'l2': l2_loss,
            'srp_reg': srp_reg
        }

class iTransformerTrainer:
    """Trainer for iTransformer model"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 1e-4,
        physics_weight: float = 1.0,
        output_dir: str = 'results_itransformer'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        # Loss function
        self.criterion = PhysicsConstrainedLoss(
            physics_weight=physics_weight,
            l2_weight=0.001
        )

        # Optimizer with cosine annealing
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,  # Total epochs
            eta_min=learning_rate * 0.1
        )

        # Tensorboard
        self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_yaw_loss': [], 'val_yaw_loss': [],
            'train_physics_loss': [], 'val_physics_loss': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0

        os.makedirs(output_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_yaw_loss = 0
        total_physics_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(inputs)

            # Calculate loss
            losses = self.criterion(predictions, targets, self.model)

            # Backward pass
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += losses['total'].item()
            total_yaw_loss += losses['yaw'].item()
            total_physics_loss += losses['physics'].item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Yaw': f"{losses['yaw'].item():.4f}",
                'Physics': f"{losses['physics'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        return {
            'loss': total_loss / len(self.train_loader),
            'yaw_loss': total_yaw_loss / len(self.train_loader),
            'physics_loss': total_physics_loss / len(self.train_loader)
        }

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_yaw_loss = 0
        total_physics_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                # Forward pass
                predictions = self.model(inputs)
                losses = self.criterion(predictions, targets, self.model)

                # Update metrics
                total_loss += losses['total'].item()
                total_yaw_loss += losses['yaw'].item()
                total_physics_loss += losses['physics'].item()

                pbar.set_postfix({
                    'Val Loss': f"{losses['total'].item():.4f}",
                    'Val Yaw': f"{losses['yaw'].item():.4f}"
                })

        return {
            'loss': total_loss / len(self.val_loader),
            'yaw_loss': total_yaw_loss / len(self.val_loader),
            'physics_loss': total_physics_loss / len(self.val_loader)
        }

    def train(self, epochs: int = 50):
        """Main training loop"""
        print(f"🚀 开始训练 iTransformer ({epochs} 轮)...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("=" * 50)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update learning rate
            self.scheduler.step()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_yaw_loss'].append(train_metrics['yaw_loss'])
            self.history['val_yaw_loss'].append(val_metrics['yaw_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_physics_loss'].append(val_metrics['physics_loss'])

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('YawLoss/Train', train_metrics['yaw_loss'], epoch)
            self.writer.add_scalar('YawLoss/Val', val_metrics['yaw_loss'], epoch)
            self.writer.add_scalar('PhysicsLoss/Train', train_metrics['physics_loss'], epoch)
            self.writer.add_scalar('PhysicsLoss/Val', val_metrics['physics_loss'], epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Yaw: {train_metrics['yaw_loss']:.4f}, Val Yaw: {val_metrics['yaw_loss']:.4f}")
            print(f"Train Physics: {train_metrics['physics_loss']:.4f}, Val Physics: {val_metrics['physics_loss']:.4f}")

            # Early stopping and best model saving
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print("💾 保存最佳模型")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # Save final model
        self.save_model('final_model.pth')

        # Plot training curves
        self.plot_training_curves()

        print(f"\n✅ 训练完成! 最佳验证损失: {self.best_val_loss:.4f}")

    def save_model(self, filename: str):
        """Save model state dict"""
        filepath = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, filepath)

    def load_model(self, filename: str):
        """Load model state dict"""
        filepath = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Total loss
        axes[0,0].plot(self.history['train_loss'], label='Train')
        axes[0,0].plot(self.history['val_loss'], label='Validation')
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Yaw loss
        axes[0,1].plot(self.history['train_yaw_loss'], label='Train')
        axes[0,1].plot(self.history['val_yaw_loss'], label='Validation')
        axes[0,1].set_title('Yaw Prediction Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)

        # Physics loss
        axes[1,0].plot(self.history['train_physics_loss'], label='Train')
        axes[1,0].plot(self.history['val_physics_loss'], label='Validation')
        axes[1,0].set_title('Physics Constraint Loss')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True)

        # Learning rate (if we track it)
        axes[1,1].set_title('Model Architecture')
        axes[1,1].text(0.1, 0.5, f'iTransformer\n- d_model: 128\n- num_heads: 8\n- num_layers: 4\n- Parameters: {sum(p.numel() for p in self.model.parameters()):,}',
                      transform=axes[1,1].transAxes, fontsize=12, verticalalignment='center')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='iTransformer训练 for BDS-3 MEO卫星')
    parser.add_argument('--data-dir', default='obx-dataset', help='OBX数据目录')
    parser.add_argument('--output-dir', default='results_itransformer', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--physics-weight', type=float, default=1.0, help='物理约束权重')
    parser.add_argument('--target-sats', default='C23,C25', help='目标卫星')
    parser.add_argument('--seq-length', type=int, default=60, help='序列长度')
    parser.add_argument('--device', default='auto', help='计算设备 (auto/cpu/cuda/mps)')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("🚀 启动 iTransformer 训练")
    print("=" * 50)
    print(f"设备: {device}")
    print(f"目标卫星: {args.target_sats}")
    print(f"序列长度: {args.seq_length}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"物理权重: {args.physics_weight}")
    print("=" * 50)

    # Setup data
    target_satellites = args.target_sats.split(',')
    data_module = SatelliteDataModule(
        data_dir=args.data_dir,
        target_sats=target_satellites,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        val_split=0.2
    )

    processed_data = data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create model
    model = iTransformerCombinedModel(
        input_dim=5,  # time, beta, mu, sat_type_cast, sat_type_secm
        transformer_d_model=128,
        transformer_num_heads=8,
        transformer_num_layers=4,
        transformer_d_ff=512,
        seq_length=args.seq_length,
        dropout=0.1
    )

    # Create trainer
    trainer = iTransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
        output_dir=args.output_dir
    )

    # Train model
    trainer.train(epochs=args.epochs)

    # Load best model for evaluation
    trainer.load_model('best_model.pth')

    print("\n🔍 模型评估...")

    # Simple evaluation (calculate RMSE on validation set)
    model.eval()
    total_yaw_error = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            predictions = model(inputs)
            yaw_pred = predictions['yaw']['prediction']
            yaw_true = targets['yaw']

            # Calculate RMSE
            error = torch.sqrt(torch.mean((yaw_pred - yaw_true) ** 2))
            total_yaw_error += error.item() * len(yaw_true)
            total_samples += len(yaw_true)

    final_rmse = total_yaw_error / total_samples
    print(f"最终验证集 偏航角RMSE: {final_rmse:.4f}°")

    # Compare with baseline
    baseline_rmse = 104.0  # From previous analysis
    improvement = (baseline_rmse - final_rmse) / baseline_rmse * 100
    print(f"相比基线改进: {improvement:+.1f}%")

    print(f"\n🎉 训练完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
