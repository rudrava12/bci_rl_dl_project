"""Centralized configuration for hyperparameters and settings."""

import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# ==========================
# LOGGING CONFIGURATION
# ==========================

def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to save logs
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Create parent directories if they don't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.NullHandler())
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=handlers
    )


# ==========================
# MODEL CONFIGURATION
# ==========================

@dataclass
class CNNConfig:
    """CNN model hyperparameters."""
    input_channels: int = 64
    num_classes: int = 3
    feature_dim: int = 64
    conv_channels: tuple = (32, 64, 128)
    kernel_size: int = 3
    dropout_rate: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 1e-5


@dataclass
class DQNConfig:
    """DQN agent hyperparameters."""
    state_size: int = 64
    action_size: int = 3
    hidden_size: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    target_update_freq: int = 1000
    buffer_capacity: int = 10000


# ==========================
# DATA CONFIGURATION
# ==========================

@dataclass
class DataConfig:
    """Data loading and preprocessing parameters."""
    sample_rate: int = 250  # Hz
    l_freq: float = 1.0  # Hz (low frequency cutoff)
    h_freq: float = 40.0  # Hz (high frequency cutoff)
    window_size: int = 128  # samples
    overlap: float = 0.5  # 50% overlap
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize: bool = True


# ==========================
# TRAINING CONFIGURATION
# ==========================

@dataclass
class TrainingConfig:
    """Training loop parameters."""
    # CNN Training
    cnn_epochs: int = 20
    cnn_batch_size: int = 32
    cnn_early_stopping_patience: int = 5
    
    # DQN Training
    dqn_episodes: int = 100
    dqn_steps_per_episode: int = 50
    dqn_batch_size: int = 32
    dqn_buffer_capacity: int = 10000
    dqn_learn_start: int = 100  # Start learning after N experiences
    
    # General
    device: str = "cuda"  # or "cpu"
    random_seed: int = 42
    num_workers: int = 4


# ==========================
# PATH CONFIGURATION
# ==========================

@dataclass
class PathConfig:
    """File paths for data and models."""
    data_dir: str = "data/eeg/files"
    model_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    cnn_model_path: str = "models/checkpoints/cnn_model.pth"
    dqn_model_path: str = "models/checkpoints/dqn_agent.pth"
    config_path: str = "config/config.py"


# ==========================
# ENVIRONMENT CONFIGURATION
# ==========================

@dataclass
class EnvConfig:
    """BCI environment parameters."""
    state_size: int = 3
    action_size: int = 3
    max_steps: int = 100
    actions: tuple = ("focus", "relax", "neutral")


# ==========================
# GLOBAL CONFIG
# ==========================

class Config:
    """Master configuration class."""
    
    cnn = CNNConfig()
    dqn = DQNConfig()
    data = DataConfig()
    training = TrainingConfig()
    paths = PathConfig()
    env = EnvConfig()
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            "cnn": cls.cnn.__dict__,
            "dqn": cls.dqn.__dict__,
            "data": cls.data.__dict__,
            "training": cls.training.__dict__,
            "paths": cls.paths.__dict__,
            "env": cls.env.__dict__,
        }
    
    @classmethod
    def print_config(cls) -> None:
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print("PROJECT CONFIGURATION")
        print("="*60)
        for section, config in cls.to_dict().items():
            print(f"\n{section.upper()}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        print("\n" + "="*60 + "\n")


# Initialize logging on import
setup_logging(log_level=logging.INFO, log_file="logs/training.log")
