#!/usr/bin/env python3
"""
PyTorch dataset and dataloader for BDS-3 MEO satellite data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from .obx_processor import OBXProcessor

class SatelliteDataset(Dataset):
    """PyTorch Dataset for satellite time series data"""
    
    def __init__(
        self,
        features: Dict,
        targets: Dict,
        seq_length: int = 60,
        prediction_length: int = 1,
        stride: int = 10  # Sample every 10th sequence to reduce data size
    ):
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        self.stride = stride
        
        # Convert data to sequences
        self.yaw_sequences = []
        self.yaw_targets = []
        self.srp_sequences = []
        
        # Process yaw data
        for sat_key in features['yaw']:
            yaw_data = np.array(features['yaw'][sat_key])
            yaw_target = np.array(targets['yaw'][sat_key])
            
            # Create sequences
            for i in range(0, len(yaw_data) - seq_length + 1, stride):
                if i + seq_length < len(yaw_data):
                    self.yaw_sequences.append(yaw_data[i:i+seq_length])
                    self.yaw_targets.append(yaw_target[i+seq_length-1])
        
        # Process SRP data (same sequences but for SRP model input)
        for sat_key in features['yaw']:  # Use same keys
            yaw_data = np.array(features['yaw'][sat_key])
            
            # For SRP, we need the same temporal sequences but different target
            for i in range(0, len(yaw_data) - seq_length + 1, stride):
                if i + seq_length < len(yaw_data):
                    self.srp_sequences.append(yaw_data[i:i+seq_length])
        
        # Convert to numpy arrays
        self.yaw_sequences = np.array(self.yaw_sequences, dtype=np.float32)
        self.yaw_targets = np.array(self.yaw_targets, dtype=np.float32)
        self.srp_sequences = np.array(self.srp_sequences, dtype=np.float32)
        
        print(f"Dataset created: {len(self.yaw_sequences)} sequences")
        print(f"Sequence shape: {self.yaw_sequences.shape}")
        
    def __len__(self):
        return len(self.yaw_sequences)
    
    def __getitem__(self, idx):
        return {
            'yaw_sequence': torch.FloatTensor(self.yaw_sequences[idx]),
            'yaw_target': torch.FloatTensor([self.yaw_targets[idx]]),
            'srp_sequence': torch.FloatTensor(self.srp_sequences[idx])
        }

class SatelliteDataModule:
    """Data module for managing training and validation data"""
    
    def __init__(
        self,
        data_dir: str,
        target_sats: List[str],
        seq_length: int = 60,
        batch_size: int = 16,
        val_split: float = 0.2,
        num_workers: int = 4
    ):
        self.data_dir = data_dir
        self.target_sats = target_sats
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        
        self.processor = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
    def setup(self):
        """Setup data processing and create datasets"""
        print("🔧 设置数据处理...")
        
        # Process OBX data
        self.processor = OBXProcessor(self.data_dir)
        processed_data = self.processor.process_dataset(target_sats=self.target_sats)
        
        # Prepare ML dataset
        features, targets = self.processor.prepare_ml_dataset(processed_data)
        
        # Create full dataset
        full_dataset = SatelliteDataset(
            features=features,
            targets=targets,
            seq_length=self.seq_length
        )
        
        # Split dataset
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
        
        return processed_data
        
    def train_dataloader(self):
        """Create training dataloader"""
        if self.train_dataset is None:
            raise RuntimeError("请先调用 setup() 方法")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for MPS compatibility
            pin_memory=False,  # Disable for MPS
            drop_last=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        if self.val_dataset is None:
            raise RuntimeError("请先调用 setup() 方法")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for MPS compatibility
            pin_memory=False,  # Disable for MPS
            drop_last=False,
            collate_fn=collate_fn
        )

def collate_fn(batch):
    """Custom collate function for batching"""
    yaw_sequences = torch.stack([item['yaw_sequence'] for item in batch])
    yaw_targets = torch.stack([item['yaw_target'] for item in batch])
    srp_sequences = torch.stack([item['srp_sequence'] for item in batch])
    
    return {
        'inputs': {
            'yaw': yaw_sequences,     # [batch_size, seq_len, features]
            'srp': srp_sequences      # [batch_size, seq_len, features]
        },
        'targets': {
            'yaw': yaw_targets.squeeze(-1)  # [batch_size]
        }
    }
