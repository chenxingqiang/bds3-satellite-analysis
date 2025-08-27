#!/usr/bin/env python3
"""
iTransformer model for BDS-3 MEO satellite yaw attitude analysis using PyTorch
Based on iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [seq_len, 1, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class VariableEmbedding(nn.Module):
    """Individual embedding for each variable/feature"""
    
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Separate embedding for each variable
        self.variable_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            ) for _ in range(input_dim)
        ])
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        embedded_vars = []
        for i in range(self.input_dim):
            var_data = x[:, :, i:i+1]  # [batch_size, seq_len, 1]
            embedded = self.variable_embeddings[i](var_data)  # [batch_size, seq_len, d_model]
            embedded_vars.append(embedded)
        
        # Stack: [batch_size, seq_len, input_dim, d_model]
        return torch.stack(embedded_vars, dim=2)

class EclipseAwareMultiHeadAttention(nn.Module):
    """Multi-head attention with eclipse period awareness"""
    
    def __init__(self, d_model: int, num_heads: int, eclipse_boost: float = 2.0, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.eclipse_boost = eclipse_boost
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, query, key, value, eclipse_mask=None, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply eclipse boost
        if eclipse_mask is not None:
            # eclipse_mask: [batch_size, seq_len]
            eclipse_boost_mask = torch.where(eclipse_mask, self.eclipse_boost, 1.0)
            eclipse_boost_mask = eclipse_boost_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores * eclipse_boost_mask
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights

class iTransformerBlock(nn.Module):
    """Single iTransformer block with eclipse awareness"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = EclipseAwareMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, eclipse_mask=None, mask=None):
        # x: [batch_size, seq_len, n_features, d_model]
        batch_size, seq_len, n_features, d_model = x.shape
        
        # Reshape for attention: [batch_size * n_features, seq_len, d_model]
        x_reshaped = x.view(batch_size * n_features, seq_len, d_model)
        
        # Expand eclipse_mask for all features if provided
        if eclipse_mask is not None:
            eclipse_mask_expanded = eclipse_mask.unsqueeze(1).expand(-1, n_features, -1)
            eclipse_mask_expanded = eclipse_mask_expanded.contiguous().view(batch_size * n_features, seq_len)
        else:
            eclipse_mask_expanded = None
        
        # Self-attention
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped, 
                                       eclipse_mask_expanded, mask)
        
        # Reshape back: [batch_size, seq_len, n_features, d_model]
        attn_output = attn_output.view(batch_size, seq_len, n_features, d_model)
        
        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class YawAttitudeiTransformer(nn.Module):
    """iTransformer model for yaw attitude prediction with physics constraints"""
    
    def __init__(
        self,
        input_dim: int = 5,  # time, beta, mu, sat_type_cast, sat_type_secm
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        seq_length: int = 60,
        dropout: float = 0.1,
        eclipse_threshold: float = 12.9
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length
        self.eclipse_threshold = eclipse_threshold
        
        # Variable embedding
        self.variable_embedding = VariableEmbedding(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_length)
        
        # iTransformer blocks
        self.transformer_blocks = nn.ModuleList([
            iTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling and output layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, d_model))  # Pool over seq_len and features
        self.output_norm = nn.LayerNorm(d_model)
        
        # Yaw prediction head
        self.yaw_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Physics constraint head
        self.physics_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Extract eclipse mask (|beta| < threshold)
        beta = x[:, :, 1]  # Second feature is beta angle
        eclipse_mask = torch.abs(beta) < self.eclipse_threshold
        
        # Variable embedding: [batch_size, seq_len, input_dim, d_model]
        embedded = self.variable_embedding(x)
        
        # Add positional encoding (need to reshape for broadcasting)
        # embedded shape: [batch_size, seq_len, input_dim, d_model]
        # We need to add positional encoding to each variable
        for i in range(self.input_dim):
            var_embedded = embedded[:, :, i, :].transpose(0, 1)  # [seq_len, batch_size, d_model]
            var_embedded = self.pos_encoding(var_embedded)  # Add positional encoding
            embedded[:, :, i, :] = var_embedded.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        
        # Apply transformer blocks
        x = embedded
        for block in self.transformer_blocks:
            x = block(x, eclipse_mask=eclipse_mask)
        
        # Global pooling: [batch_size, seq_len, n_features, d_model] -> [batch_size, d_model]
        # Average over sequence length and features
        pooled = torch.mean(x, dim=(1, 2))  # [batch_size, d_model]
        pooled = self.output_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Output predictions
        yaw_pred = self.yaw_head(pooled).squeeze(-1)  # [batch_size]
        theoretical_rate = self.physics_head(pooled).squeeze(-1)  # [batch_size]
        
        # Aggregate eclipse mask - True if any timestep is in eclipse
        eclipse_flag = torch.any(eclipse_mask, dim=1)  # [batch_size]
        
        return {
            'prediction': yaw_pred,
            'theoretical_rate': theoretical_rate,
            'eclipse_mask': eclipse_flag
        }

class SRPCompensationModel(nn.Module):
    """Simplified SRP compensation model using PyTorch"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dims: dict = None):
        super().__init__()
        
        if output_dims is None:
            output_dims = {'d_params': 4, 'y_params': 2, 'b_params': 2}
        
        self.input_dim = input_dim
        self.output_dims = output_dims
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Parameter-specific heads
        self.d_head = nn.Linear(hidden_dim, output_dims['d_params'])
        self.y_head = nn.Linear(hidden_dim, output_dims['y_params'])
        self.b_head = nn.Linear(hidden_dim, output_dims['b_params'])
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> take last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch_size, input_dim]
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate parameters
        d_params = self.d_head(features)
        y_params = self.y_head(features)
        b_params = self.b_head(features)
        
        return {
            'd_params': d_params,
            'y_params': y_params,
            'b_params': b_params
        }

class iTransformerCombinedModel(nn.Module):
    """Combined model with iTransformer for yaw attitude and simplified SRP compensation"""
    
    def __init__(
        self,
        input_dim: int = 5,
        transformer_d_model: int = 128,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 4,
        transformer_d_ff: int = 512,
        seq_length: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.yaw_model = YawAttitudeiTransformer(
            input_dim=input_dim,
            d_model=transformer_d_model,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            d_ff=transformer_d_ff,
            seq_length=seq_length,
            dropout=dropout
        )
        
        self.srp_model = SRPCompensationModel(
            input_dim=input_dim,
            hidden_dim=64
        )
        
    def forward(self, inputs):
        """
        Forward pass through both models
        
        Args:
            inputs: Dictionary with keys 'yaw' and 'srp'
                - inputs['yaw']: [batch_size, seq_len, features] for yaw model
                - inputs['srp']: [batch_size, seq_len, features] for SRP model
        """
        yaw_outputs = self.yaw_model(inputs['yaw'])
        srp_outputs = self.srp_model(inputs['srp'])
        
        return {
            'yaw': yaw_outputs,
            'srp': srp_outputs
        }
