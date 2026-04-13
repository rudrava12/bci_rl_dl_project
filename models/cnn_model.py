"""Convolutional Neural Network for EEG classification and feature extraction."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class EEG_CNN(nn.Module):
    """
    CNN architecture for EEG processing.
    
    - Input: (batch, 64 channels, 128 samples)
    - Output: Classification logits (batch, 3) and features (batch, 64)
    """
    
    def __init__(
        self, 
        input_channels: int = 64,
        num_classes: int = 3,
        feature_dim: int = 64
    ):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of EEG channels (default: 64)
            num_classes: Number of output classes (default: 3)
            feature_dim: Dimension of feature vector (default: 64)
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 10, feature_dim)
        self.bn_fc1 = nn.BatchNorm1d(feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"EEG_CNN initialized: {input_channels} channels -> {num_classes} classes")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 64, 128)
            
        Returns:
            Tuple of (logits, features)
                logits: Classification logits (batch, 3)
                features: Feature representation (batch, 64)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with feature extraction
        features = self.fc1(x)
        features = self.bn_fc1(features)
        features = F.relu(features)
        features = self.dropout(features)
        
        # Output
        logits = self.fc2(features)
        
        return logits, features
