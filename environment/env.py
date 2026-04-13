"""BCI Environment for RL training with EEG-based feedback."""

from typing import Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BCIEnv:
    """
    Brain-Computer Interface Environment for DQN training.
    
    State space: Continuous features from CNN (64-dim)
    Action space: {0: focus, 1: relax, 2: neutral}
    Reward: Based on EEG signal quality and action appropriateness
    """
    
    def __init__(self, state_size: int = 3):
        """
        Initialize BCI environment.
        
        Args:
            state_size: Dimension of state space (default: 3 for demo)
        """
        self.state_size = state_size
        self.state = np.random.rand(state_size)
        self.step_count = 0
        self.max_steps = 100
        self.target_state = np.random.rand(state_size)  # Initialize with random target
        logger.info(f"BCIEnv initialized with state_size={state_size}")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state array
        """
        self.state = np.random.rand(self.state_size)
        self.step_count = 0
        # Random target state
        self.target_state = np.random.rand(self.state_size)
        logger.debug(f"Environment reset")
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step of the environment with the given action.
        
        Args:
            action: Action index (0, 1, or 2)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action: {action}")
        
        self.step_count += 1
        
        # Reward design: action 0 is best (+1), action 1 is neutral (0), action 2 is worst (-1)
        reward = self._compute_reward(action)
        
        # Update state based on action (simulate BCI dynamics)
        self.state = self._update_state(action)
        
        done = self.step_count >= self.max_steps
        
        return self.state.copy(), reward, done
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward based on action and current state.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        # Baseline reward
        base_reward = {0: 1.0, 1: 0.0, 2: -1.0}[action]
        
        # Bonus for proximity to target state
        state_distance = np.linalg.norm(self.state - self.target_state)
        proximity_bonus = -state_distance * 0.5
        
        # Step penalty to encourage faster learning
        step_penalty = -0.01 * self.step_count / self.max_steps
        
        return base_reward + proximity_bonus + step_penalty
    
    def _update_state(self, action: int) -> np.ndarray:
        """
        Update state based on action.
        
        Args:
            action: Action taken
            
        Returns:
            New state
        """
        new_state = self.state.copy()
        
        # Action-dependent dynamics
        if action == 0:  # Focus
            new_state += np.random.randn(self.state_size) * 0.1
        elif action == 1:  # Relax
            new_state -= np.random.randn(self.state_size) * 0.05
        else:  # Neutral
            new_state += np.random.randn(self.state_size) * 0.2
        
        # Clip to [0, 1]
        new_state = np.clip(new_state, 0, 1)
        return new_state
    
    def render(self) -> None:
        """Render current state (for debugging)."""
        print(f"Step: {self.step_count} | State: {self.state}")
