"""
Experience buffer for reinforcement learning.

This module implements an experience replay buffer using
the classic design from DQN (Mnih et al., 2015).

The buffer stores transitions (state, action, reward, next_state)
and allows random sampling for training stability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch

from koikoi.core.constants import Action


class Experience(NamedTuple):
    """
    A single experience/transition tuple.
    
    Attributes:
        state: Game state feature tensor
        action: Action taken (format depends on action type)
        reward: Reward received
        action_type: Type of action ('discard', 'pick', 'koikoi')
        action_mask: Valid action mask at time of decision
        next_state: Resulting game state (optional)
        done: Whether episode ended
    """
    state: np.ndarray
    action: Action
    reward: float
    action_type: str
    action_mask: np.ndarray
    next_state: Optional[np.ndarray] = None
    done: bool = False


T = TypeVar('T')


@dataclass
class ExperienceBuffer:
    """
    Replay buffer for storing and sampling experiences.
    
    This implements a ring buffer that overwrites oldest
    experiences when capacity is reached.
    
    The buffer is partitioned by action type, allowing
    balanced sampling across different decision types
    (discard, pick, koi-koi).
    
    Attributes:
        capacity: Maximum number of experiences per action type
        
    Design Notes:
        - Uses separate buffers for each action type to ensure
          balanced training across all decision types
        - Implements ring buffer for O(1) insertion
        - Supports both uniform and weighted sampling
    
    Example:
        >>> buffer = ExperienceBuffer(capacity=10000)
        >>> buffer.push(experience)
        >>> batch = buffer.sample('discard', batch_size=32)
    """
    
    capacity: int = 10000
    
    # Separate buffers for each action type
    _buffers: dict = field(default_factory=lambda: {
        'discard': [],
        'pick': [],
        'koikoi': [],
    })
    _positions: dict = field(default_factory=lambda: {
        'discard': 0,
        'pick': 0,
        'koikoi': 0,
    })
    
    def push(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Uses ring buffer semantics - overwrites oldest
        experience when capacity is reached.
        
        Args:
            experience: Experience tuple to store
        """
        action_type = experience.action_type
        buffer = self._buffers[action_type]
        position = self._positions[action_type]
        
        if len(buffer) < self.capacity:
            buffer.append(experience)
        else:
            buffer[position] = experience
        
        self._positions[action_type] = (position + 1) % self.capacity
    
    def push_many(self, experiences: List[Experience]) -> None:
        """
        Add multiple experiences to the buffer.
        
        Args:
            experiences: List of experience tuples
        """
        for exp in experiences:
            self.push(exp)
    
    def sample(
        self,
        action_type: str,
        batch_size: int,
    ) -> Tuple[torch.Tensor, List[Action], torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.
        
        Args:
            action_type: Type of action to sample ('discard', 'pick', 'koikoi')
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, masks):
                - states: FloatTensor of shape (batch_size, feature_dim)
                - actions: List of actions
                - rewards: FloatTensor of shape (batch_size,)
                - masks: FloatTensor of shape (batch_size, num_actions)
                
        Raises:
            ValueError: If buffer has fewer than batch_size experiences
        """
        buffer = self._buffers[action_type]
        
        if len(buffer) < batch_size:
            raise ValueError(
                f"Buffer has only {len(buffer)} experiences, "
                f"but batch_size is {batch_size}"
            )
        
        batch = random.sample(buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor([e.reward for e in batch])
        masks = torch.FloatTensor(np.array([e.action_mask for e in batch]))
        
        return states, actions, rewards, masks
    
    def sample_all(
        self,
        action_type: str,
    ) -> Tuple[torch.Tensor, List[Any], torch.Tensor, torch.Tensor]:
        """
        Get all experiences of a specific action type.
        
        Useful for Monte-Carlo style updates where we process
        all experiences from a completed episode.
        
        Args:
            action_type: Type of action to retrieve
            
        Returns:
            Same format as sample()
        """
        buffer = self._buffers[action_type]
        
        if not buffer:
            # Return empty tensors
            return (
                torch.FloatTensor([]),
                [],
                torch.FloatTensor([]),
                torch.FloatTensor([]),
            )
        
        states = torch.FloatTensor(np.array([e.state for e in buffer]))
        actions = [e.action for e in buffer]
        rewards = torch.FloatTensor([e.reward for e in buffer])
        masks = torch.FloatTensor(np.array([e.action_mask for e in buffer]))
        
        return states, actions, rewards, masks
    
    def clear(self, action_type: Optional[str] = None) -> None:
        """
        Clear the buffer.
        
        Args:
            action_type: If specified, only clear that action type.
                If None, clear all buffers.
        """
        if action_type is None:
            for key in self._buffers:
                self._buffers[key] = []
                self._positions[key] = 0
        else:
            self._buffers[action_type] = []
            self._positions[action_type] = 0
    
    def __len__(self) -> int:
        """Return total number of experiences across all action types."""
        return sum(len(b) for b in self._buffers.values())
    
    def size(self, action_type: str) -> int:
        """
        Get size of a specific action type buffer.
        
        Args:
            action_type: Action type to check
            
        Returns:
            Number of experiences stored
        """
        return len(self._buffers[action_type])
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for training.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if all buffers have at least batch_size experiences
        """
        return all(len(b) >= batch_size for b in self._buffers.values())


@dataclass
class RolloutBuffer:
    """
    Buffer for storing complete rollouts (trajectories).
    
    Unlike ExperienceBuffer, this stores entire episodes
    together, which is useful for:
    - Monte-Carlo returns calculation
    - GAE (Generalized Advantage Estimation)
    - Analyzing complete game outcomes
    
    Attributes:
        experiences: List of experience lists (one per episode)
        max_episodes: Maximum number of episodes to store
    """
    
    max_episodes: int = 100
    experiences: List[List[Experience]] = field(default_factory=list)
    _current_episode: List[Experience] = field(default_factory=list)
    
    def step(self, experience: Experience) -> None:
        """
        Record a single step within current episode.
        
        Args:
            experience: Experience from this timestep
        """
        self._current_episode.append(experience)
    
    def end_episode(self, final_reward: float) -> None:
        """
        Mark end of episode and compute returns.
        
        The final_reward is propagated back through all
        experiences in the episode (Monte-Carlo style).
        
        Args:
            final_reward: Final reward for this episode
        """
        # Update all experiences with final reward
        # For MC learning, all steps get the same final reward
        updated_episode = []
        for exp in self._current_episode:
            updated_exp = Experience(
                state=exp.state,
                action=exp.action,
                reward=final_reward,  # MC: use final reward
                action_type=exp.action_type,
                action_mask=exp.action_mask,
                next_state=exp.next_state,
                done=True,
            )
            updated_episode.append(updated_exp)
        
        self.experiences.append(updated_episode)
        self._current_episode = []
        
        # Maintain max episodes
        while len(self.experiences) > self.max_episodes:
            self.experiences.pop(0)
    
    def get_all_experiences(self) -> List[Experience]:
        """
        Get all experiences from all stored episodes.
        
        Returns:
            Flat list of all experiences
        """
        result = []
        for episode in self.experiences:
            result.extend(episode)
        return result
    
    def clear(self) -> None:
        """Clear all stored episodes."""
        self.experiences = []
        self._current_episode = []
    
    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.experiences)
