"""Experience replay buffer for DQN training."""

from typing import List, Tuple
from collections import deque
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling training experiences.
    
    Stores transitions: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        logger.info(f"ReplayBuffer initialized with capacity={capacity}")
    
    def store(self, experience: Tuple) -> None:
        """
        Store a single experience in the buffer.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
            
        Raises:
            ValueError: If batch_size exceeds buffer size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of stored experiences
        """
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """
        Check if buffer is at capacity.
        
        Returns:
            True if buffer is full
        """
        return len(self.buffer) == self.capacity
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        logger.info("ReplayBuffer cleared")
