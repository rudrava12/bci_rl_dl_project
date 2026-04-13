"""Main entry point for BCI system training and evaluation."""

import torch
import numpy as np
import os
import logging
from pathlib import Path

from models.cnn_model import EEG_CNN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
from environment.env import BCIEnv
from preprocessing.eeg_loader import load_eeg, segment_data, normalize_eeg
from config.config import Config, setup_logging

logger = logging.getLogger(__name__)


class BCISystem:
    """Integrated BCI system combining CNN feature extraction and DQN control."""
    
    def __init__(self, config: Config):
        """Initialize BCI system."""
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.cnn = EEG_CNN().to(self.device)
        self.agent = DQNAgent(
            state_size=config.dqn.state_size,
            action_size=config.dqn.action_size,
            learning_rate=config.dqn.learning_rate
        )
        self.buffer = ReplayBuffer(capacity=config.dqn.buffer_capacity)
        self.env = BCIEnv(state_size=config.env.state_size)
        
        # Load pre-trained CNN if available
        self._load_cnn()
    
    def _load_cnn(self):
        """Load pre-trained CNN model."""
        try:
            self.cnn.load_state_dict(
                torch.load(self.config.paths.cnn_model_path, map_location=self.device)
            )
            self.cnn.eval()
            logger.info(f"CNN model loaded from {self.config.paths.cnn_model_path}")
        except FileNotFoundError:
            logger.warning(f"CNN model not found at {self.config.paths.cnn_model_path}")
            logger.info("Initializing CNN with random weights")
    
    def load_real_eeg_segments(self, base_path: str = "data/eeg/files", max_files: int = 5):
        """Load real EEG data segments."""
        logger.info(f"Loading EEG data from {base_path}...")
        all_segments = []
        
        files_loaded = 0
        for subject in os.listdir(base_path):
            subject_path = os.path.join(base_path, subject)
            
            if not os.path.isdir(subject_path):
                continue
            
            for file in os.listdir(subject_path):
                if file.endswith(".edf"):
                    try:
                        file_path = os.path.join(subject_path, file)
                        logger.info(f"Loading: {file_path}")
                        
                        raw = load_eeg(file_path)
                        segments = segment_data(raw, window_size=self.config.data.window_size)
                        
                        if self.config.data.normalize:
                            segments = normalize_eeg(segments)
                        
                        all_segments.append(segments)
                        files_loaded += 1
                        
                        if files_loaded >= max_files:
                            break
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")
                        continue
            
            if files_loaded >= max_files:
                break
        
        if not all_segments:
            logger.warning("No EEG data found. Using random features.")
            return None
        
        result = np.concatenate(all_segments, axis=0)
        logger.info(f"Loaded {len(result)} EEG segments")
        return result
    
    def train_integrated(self, num_episodes: int = 50):
        """
        Train integrated CNN-DQN system.
        CNN extracts features from EEG, DQN learns control policy.
        """
        logger.info("Starting integrated CNN-DQN training...")
        
        # Load real EEG data if available
        eeg_data = self.load_real_eeg_segments()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.training.dqn_steps_per_episode):
                # Get CNN features if EEG data available
                if eeg_data is not None:
                    idx = np.random.randint(0, len(eeg_data))
                    eeg_segment = eeg_data[idx:idx+1]
                    
                    eeg_tensor = torch.FloatTensor(eeg_segment).to(self.device)
                    
                    with torch.no_grad():
                        _, features = self.cnn(eeg_tensor)
                    
                    state = features.cpu().numpy()[0]
                
                # Choose action
                action = self.agent.choose_action(state)
                
                # Get next state from EEG if available
                if eeg_data is not None:
                    idx = np.random.randint(0, len(eeg_data))
                    next_eeg_segment = eeg_data[idx:idx+1]
                    
                    next_tensor = torch.FloatTensor(next_eeg_segment).to(self.device)
                    
                    with torch.no_grad():
                        _, next_features = self.cnn(next_tensor)
                    
                    next_state = next_features.cpu().numpy()[0]
                else:
                    next_state, _, _ = self.env.step(action)
                
                # Get reward from environment
                _, reward, done = self.env.step(action)
                
                # Store experience
                self.buffer.store((state, action, reward, next_state, float(done)))
                episode_reward += reward
                
                # Train if buffer ready
                if self.buffer.size() >= self.config.training.dqn_batch_size:
                    batch = self.buffer.sample(self.config.training.dqn_batch_size)
                    self.agent.train(batch)
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg (10 eps): {avg_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )
        
        # Save models
        self._save_models()
        
        logger.info("Training complete!")
        return episode_rewards
    
    def _save_models(self):
        """Save trained models."""
        Path(self.config.paths.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save CNN
        torch.save(self.cnn.state_dict(), self.config.paths.cnn_model_path)
        logger.info(f"CNN saved to {self.config.paths.cnn_model_path}")
        
        # Save DQN agent
        self.agent.save(self.config.paths.dqn_model_path)
        logger.info(f"DQN agent saved to {self.config.paths.dqn_model_path}")
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate trained system."""
        logger.info(f"Evaluating system for {num_episodes} episodes...")
        
        self.cnn.eval()
        self.agent.epsilon = 0  # No exploration
        
        # Load real EEG data if available
        eeg_data = self.load_real_eeg_segments()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.training.dqn_steps_per_episode):
                # Get CNN features if EEG data available, otherwise use env state
                if eeg_data is not None:
                    idx = np.random.randint(0, len(eeg_data))
                    eeg_segment = eeg_data[idx:idx+1]
                    
                    eeg_tensor = torch.FloatTensor(eeg_segment).to(self.device)
                    
                    with torch.no_grad():
                        _, features = self.cnn(eeg_tensor)
                    
                    state = features.cpu().numpy()[0]
                
                action = self.agent.choose_action(state)
                state, reward, done = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            logger.info(f"Eval Episode {episode+1}: Reward = {episode_reward:.2f}")
        
        avg_reward = np.mean(episode_rewards)
        logger.info(f"Average reward: {avg_reward:.2f}")
        return episode_rewards


def main():
    """Main entry point."""
    setup_logging()
    config = Config()
    config.print_config()
    
    # Create BCI system
    bci_system = BCISystem(config)
    
    # Train
    rewards = bci_system.train_integrated(num_episodes=config.training.dqn_episodes)
    
    # Evaluate
    eval_rewards = bci_system.evaluate(num_episodes=10)
    
    logger.info("BCI system training and evaluation complete!")


if __name__ == "__main__":
    main()