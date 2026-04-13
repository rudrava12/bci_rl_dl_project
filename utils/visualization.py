"""Visualization utilities for BCI system."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_learning_curve(
    rewards: List[float],
    title: str = "Learning Curve",
    window: int = 10
) -> plt.Figure:
    """
    Plot learning curve with moving average.
    
    Args:
        rewards: List of episode rewards
        title: Plot title
        window: Moving average window
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raw rewards
    ax.plot(rewards, alpha=0.3, label="Episode Reward")
    
    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg, linewidth=2, label=f"Moving Avg ({window})")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_action_distribution(
    actions: List[int],
    action_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot distribution of actions taken.
    
    Args:
        actions: List of action indices
        action_names: Names for each action
        
    Returns:
        Matplotlib figure
    """
    if action_names is None:
        action_names = [f"Action {i}" for i in range(max(actions) + 1)]
    
    counts = np.bincount(actions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(action_names[:len(counts)], counts)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Action Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_metrics_comparison(
    metrics_dict: dict,
    title: str = "Metrics Comparison"
) -> plt.Figure:
    """
    Plot comparison of multiple metrics.
    
    Args:
        metrics_dict: Dictionary of {metric_name: values}
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=(15, 5))
    
    if len(metrics_dict) == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics_dict.items()):
        ax.plot(values, linewidth=2, marker='o')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_eeg_segment(
    eeg_data: np.ndarray,
    channels: Optional[List[int]] = None,
    title: str = "EEG Segment"
) -> plt.Figure:
    """
    Plot EEG data segment.
    
    Args:
        eeg_data: EEG data of shape (n_channels, n_samples)
        channels: Specific channels to plot (default: first 8)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if channels is None:
        channels = list(range(min(8, eeg_data.shape[0])))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, ch in enumerate(channels):
        if ch < eeg_data.shape[0]:
            ax.plot(eeg_data[ch], alpha=0.7, label=f"Ch {ch}")
    
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names for each class
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importances: Feature importance scores
        feature_names: Names for each feature
        
    Returns:
        Matplotlib figure
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_k = min(20, len(indices))  # Show top 20
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(top_k), importances[indices[:top_k]])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_k]])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance (Top 20)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Example usage
    rewards = np.cumsum(np.random.randn(100)) + 50
    fig = plot_learning_curve(rewards.tolist())
    plt.show()
    
    actions = np.random.choice([0, 1, 2], size=100)
    fig = plot_action_distribution(actions.tolist(), ["Focus", "Relax", "Neutral"])
    plt.show()
