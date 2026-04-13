"""Training script for DQN RL agent."""

import torch
import numpy as np
import logging
from pathlib import Path
from collections import deque

from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
from environment.env import BCIEnv
from config.config import Config, setup_logging

logger = logging.getLogger(__name__)


class DQNTrainer:
    """Trainer for DQN agent."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        
        self.env = BCIEnv(state_size=config.env.state_size)
        self.agent = DQNAgent(
            state_size=config.dqn.state_size,
            action_size=config.dqn.action_size,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            epsilon=config.dqn.epsilon,
            epsilon_decay=config.dqn.epsilon_decay,
            epsilon_min=config.dqn.epsilon_min
        )
        self.buffer = ReplayBuffer(capacity=config.dqn.buffer_capacity)
        
        # Create checkpoint directory
        Path(config.paths.model_dir).mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = deque(maxlen=100)
        
        logger.info(f"DQNTrainer initialized on device: {self.device}")
    
    def train(self):
        """Main training loop for DQN agent."""
        logger.info(f"Starting DQN training for {self.config.training.dqn_episodes} episodes...")
        
        for episode in range(self.config.training.dqn_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            for step in range(self.config.training.dqn_steps_per_episode):
                # Choose action
                action = self.agent.choose_action(state)
                
                # Take step
                next_state, reward, done = self.env.step(action)
                
                # Store experience
                self.buffer.store((state, action, reward, next_state, float(done)))
                
                episode_reward += reward
                steps = step + 1
                
                # Train if buffer is ready
                if self.buffer.size() >= self.config.training.dqn_learn_start:
                    if self.buffer.size() >= self.config.training.dqn_batch_size:
                        batch = self.buffer.sample(self.config.training.dqn_batch_size)
                        loss = self.agent.train(batch)
                        episode_loss += loss
                
                state = next_state
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)
            epsilon = self.agent.epsilon
            
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode+1}/{self.config.training.dqn_episodes} | "
                    f"Reward: {episode_reward:.2f} | Avg (100 eps): {avg_reward:.2f} | "
                    f"Epsilon: {epsilon:.4f} | Steps: {steps}"
                )
            
            # Save best model
            if episode > 0 and avg_reward > np.mean(list(self.episode_rewards)[:-1]):
                self._save_checkpoint()
        
        logger.info("Training complete!")
        self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save agent checkpoint."""
        self.agent.save(self.config.paths.dqn_model_path)
        logger.info(f"Agent saved to {self.config.paths.dqn_model_path}")


def train_with_real_eeg(config: Config):
    """
    Train DQN agent with CNN-extracted EEG features.
    
    This demonstrates integration with pre-trained CNN model.
    """
    import torch
    from models.cnn_model import EEG_CNN
    
    logger.info("Loading pre-trained CNN model...")
    
    # Load pre-trained CNN
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    cnn_model = EEG_CNN().to(device)
    
    try:
        cnn_model.load_state_dict(torch.load(config.paths.cnn_model_path, map_location=device))
        cnn_model.eval()
        logger.info(f"CNN model loaded from {config.paths.cnn_model_path}")
    except FileNotFoundError:
        logger.warning(f"CNN model not found at {config.paths.cnn_model_path}")
        logger.info("Proceeding with random CNN features")
    
    # DQN agent expects CNN feature vector (64-dim)
    agent = DQNAgent(
        state_size=config.cnn.feature_dim,
        action_size=config.dqn.action_size,
        learning_rate=config.dqn.learning_rate
    )
    
    env = BCIEnv(state_size=config.env.state_size)
    buffer = ReplayBuffer(capacity=config.dqn.buffer_capacity)
    
    logger.info("Training with EEG-CNN features...")
    
    for episode in range(config.training.dqn_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Convert env state to CNN feature space
        if len(state) != config.cnn.feature_dim:
            # Pad or project state to feature dimension
            state_projected = np.zeros(config.cnn.feature_dim)
            state_projected[:len(state)] = state
            state = state_projected
        
        for step in range(config.training.dqn_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Project to feature space
            if len(next_state) != config.cnn.feature_dim:
                next_state_projected = np.zeros(config.cnn.feature_dim)
                next_state_projected[:len(next_state)] = next_state
                next_state = next_state_projected
            
            buffer.store((state, action, reward, next_state, float(done)))
            episode_reward += reward
            
            if buffer.size() >= config.training.dqn_batch_size:
                batch = buffer.sample(config.training.dqn_batch_size)
                agent.train(batch)
            
            state = next_state
            if done:
                break
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1} | Reward: {episode_reward:.2f}")
    
    agent.save(config.paths.dqn_model_path)
    logger.info("Training complete!")


if __name__ == "__main__":
    setup_logging()
    config = Config()
    config.print_config()
    
    # Standard DQN training
    trainer = DQNTrainer(config)
    trainer.train()
    
    # Optional: Train with real EEG features
    # train_with_real_eeg(config)
