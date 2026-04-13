"""Deep Q-Network agent for RL-based BCI control."""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """Deep Q-Network neural network."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize DQN network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = self.fc1(x)
        if x.dim() > 1:  # Batch processing
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if x.dim() > 1:  # Batch processing
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Network agent for BCI control.
    
    Implements epsilon-greedy exploration and experience replay.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of actions
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Networks
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.train_steps = 0
        self.target_update_freq = 1000
        
        logger.info(
            f"DQNAgent initialized: state_size={state_size}, "
            f"action_size={action_size}, lr={learning_rate}"
        )
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Action index
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        self.model.eval()  # Set to eval mode (disables batch norm statistics computation)
        state_tensor = torch.FloatTensor(state).to(device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return torch.argmax(q_values[0]).item()
    
    def train(self, batch: List[Tuple]) -> float:
        """
        Train the DQN on a batch of experiences.
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            Loss value
        """
        self.model.train()  # Set to training mode for batch norm statistics
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (use target model in eval mode)
        self.target_model.eval()
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logger.debug(f"Target network updated at step {self.train_steps}")
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """
        Save agent model to file.
        
        Args:
            path: File path to save model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load agent model from file.
        
        Args:
            path: File path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.target_model.load_state_dict(self.model.state_dict())
        logger.info(f"Agent loaded from {path}")
