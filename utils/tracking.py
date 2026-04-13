"""Utilities for experiment tracking and visualization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments and save results."""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.config = {}
        self.timestamp = datetime.now().isoformat()
        
        logger.info(f"Experiment tracker initialized: {experiment_name}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        self.config = config
        logger.info(f"Configuration logged for {self.experiment_name}")
    
    def log_metric(self, metric_name: str, value: float, step: int = None) -> None:
        """Log a metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        entry = {"value": value, "step": step, "timestamp": datetime.now().isoformat()}
        self.metrics[metric_name].append(entry)
    
    def log_metrics(self, metrics_dict: Dict[str, float], step: int = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def save(self) -> str:
        """Save experiment results to JSON."""
        exp_data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "metrics": self.metrics
        }
        
        filename = self.log_dir / f"{self.experiment_name}_{self.timestamp.replace(':', '-')}.json"
        
        with open(filename, 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        logger.info(f"Experiment saved to {filename}")
        return str(filename)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "num_metrics": len(self.metrics),
            "metrics_summary": {}
        }
        
        for metric_name, values in self.metrics.items():
            vals = [v["value"] for v in values]
            summary["metrics_summary"][metric_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "final": float(vals[-1]) if vals else None
            }
        
        return summary


class MetricsCollector:
    """Collect and aggregate metrics during training."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.epoch_metrics = {}
        self.global_metrics = {}
    
    def record_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Record metrics for a single epoch."""
        self.epoch_metrics[epoch] = metrics
    
    def record_global(self, name: str, value: float) -> None:
        """Record a global metric."""
        if name not in self.global_metrics:
            self.global_metrics[name] = []
        self.global_metrics[name].append(value)
    
    def get_running_avg(self, name: str, window: int = 10) -> float:
        """Get running average of a metric."""
        if name not in self.global_metrics:
            return 0.0
        
        vals = self.global_metrics[name]
        if len(vals) == 0:
            return 0.0
        
        return float(np.mean(vals[-window:]))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {
            "num_epochs": len(self.epoch_metrics),
            "total_metrics_recorded": sum(len(m) for m in self.epoch_metrics.values()),
            "global_metrics": {}
        }
        
        for name, vals in self.global_metrics.items():
            if vals:
                summary["global_metrics"][name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals))
                }
        
        return summary


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.timings = {}
        self.counters = {}
    
    def start(self, name: str) -> None:
        """Start timing a process."""
        import time
        self.timings[name] = {"start": time.time(), "end": None}
    
    def end(self, name: str) -> float:
        """End timing a process and return elapsed time."""
        import time
        if name not in self.timings:
            return 0.0
        
        self.timings[name]["end"] = time.time()
        elapsed = self.timings[name]["end"] - self.timings[name]["start"]
        logger.debug(f"Timing {name}: {elapsed:.4f}s")
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def reset(self) -> None:
        """Reset all timings and counters."""
        self.timings = {}
        self.counters = {}


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker("demo_experiment")
    
    tracker.log_config({"learning_rate": 0.001, "batch_size": 32})
    
    for epoch in range(10):
        tracker.log_metric("loss", 1.0 / (epoch + 1), step=epoch)
        tracker.log_metric("accuracy", 0.5 + 0.05 * epoch, step=epoch)
    
    print("\nExperiment Summary:")
    print(json.dumps(tracker.get_summary(), indent=2))
    
    save_path = tracker.save()
    print(f"\nSaved to: {save_path}")
