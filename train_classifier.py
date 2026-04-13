"""Training script for EEG CNN classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.cnn_model import EEG_CNN
from preprocessing.eeg_loader import load_eeg, segment_data, normalize_eeg
from config.config import Config, setup_logging

logger = logging.getLogger(__name__)


class EEGClassifierTrainer:
    """Trainer for EEG CNN classifier."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        self.model = EEG_CNN(
            input_channels=config.cnn.input_channels,
            num_classes=config.cnn.num_classes,
            feature_dim=config.cnn.feature_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.cnn.learning_rate,
            weight_decay=config.cnn.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Create checkpoint directory
        Path(config.paths.model_dir).mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def load_data(self, base_path: str = "data/eeg/files", max_subjects: int = 5):
        """Load and preprocess EEG data."""
        logger.info(f"Loading EEG data from {base_path}...")
        
        all_segments = []
        all_labels = []
        
        files_processed = 0
        
        for subject in os.listdir(base_path):
            subject_path = os.path.join(base_path, subject)
            
            if not os.path.isdir(subject_path):
                continue
            
            for file in os.listdir(subject_path):
                if file.endswith(".edf"):
                    try:
                        file_path = os.path.join(subject_path, file)
                        logger.info(f"Processing: {file_path}")
                        
                        # Load and segment data
                        raw = load_eeg(file_path)
                        segments = segment_data(
                            raw,
                            window_size=self.config.data.window_size,
                            overlap=self.config.data.overlap
                        )
                        
                        # Get label
                        label = self._get_label_from_file(file_path)
                        labels = np.full(len(segments), label)
                        
                        # Normalize
                        if self.config.data.normalize:
                            segments = normalize_eeg(segments)
                        
                        all_segments.append(segments)
                        all_labels.append(labels)
                        
                        files_processed += 1
                        if files_processed >= max_subjects:
                            break
                    
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
                        continue
            
            if files_processed >= max_subjects:
                break
        
        if not all_segments:
            logger.error("No EEG data loaded!")
            raise RuntimeError("Failed to load EEG data")
        
        # Concatenate all data
        X = np.concatenate(all_segments, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def _get_label_from_file(self, file_path: str) -> int:
        """Extract label from filename."""
        if "R01" in file_path or "baseline" in file_path.lower():
            return 0  # Baseline
        elif "R02" in file_path or "R03" in file_path or "motor" in file_path.lower():
            return 1  # Motor task
        else:
            return 2  # Other
    
    def train(self):
        """Main training loop."""
        logger.info("Starting CNN classifier training...")
        
        # Load data
        try:
            X, y = self.load_data()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.config.training.random_seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.config.training.random_seed
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.cnn_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.cnn_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.cnn_batch_size)
        
        # Training loop
        for epoch in range(self.config.training.cnn_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.cnn_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.cnn_early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Test
        test_loss, test_acc, metrics = self._test(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        
        logger.info("Training complete!")
    
    def _train_epoch(self, loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            logits, _ = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate(self, loader) -> tuple:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _test(self, loader) -> tuple:
        """Test model and compute metrics."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return avg_loss, accuracy, metrics
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), self.config.paths.cnn_model_path)
        logger.info(f"Model saved to {self.config.paths.cnn_model_path}")


if __name__ == "__main__":
    setup_logging()
    config = Config()
    config.print_config()
    
    trainer = EEGClassifierTrainer(config)
    trainer.train()