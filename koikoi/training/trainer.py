"""
Training utilities for Koi-Koi AI.

This module provides high-level training interfaces including:
- Trainer class for managing the training loop
- Arena class for model evaluation
- Utility functions for common training operations

The design follows a clean separation between data collection
(TraceSimulator), storage (ExperienceBuffer), and training (Trainer).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from koikoi.ai.agent import KoiKoiAgent
    from koikoi.training.buffer import ExperienceBuffer


def time_str() -> str:
    """Return current time as formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def print_log(message: str, log_path: Optional[str] = None) -> None:
    """
    Print message and optionally write to log file.
    
    Args:
        message: Message to print/log
        log_path: Path to log file (if None, only prints)
    """
    print(message)
    if log_path:
        with open(log_path, 'a') as f:
            print(message, file=f)


@dataclass
class TrainingConfig:
    """
    Configuration for training runs.
    
    Attributes:
        batch_size: Number of samples per training batch
        learning_rate: Optimizer learning rate
        device: Device for training ('cpu' or 'cuda:0')
        log_path: Path for training logs
        save_dir: Directory for saving models
        n_loop_action_net_update: Loops between action network updates
        n_loop_arena_test: Loops between arena tests
    """
    batch_size: int = 256
    learning_rate: float = 0.0001
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_path: str = 'training_log.txt'
    save_dir: str = 'model_rl'
    n_loop_action_net_update: int = 5
    n_loop_arena_test: int = 5
    

@dataclass
class Trainer:
    """
    High-level trainer for Koi-Koi AI models.
    
    Manages the training loop including:
    - Value network optimization
    - Action network updates (periodic copy from value net)
    - Arena testing and model saving
    
    The trainer implements a DQN-style training loop with
    target network (action_net) and online network (value_net).
    
    Attributes:
        value_nets: Dictionary of value networks by action type
        action_nets: Dictionary of action networks (targets)
        config: Training configuration
        
    Example:
        >>> trainer = Trainer(value_nets, action_nets, config)
        >>> trainer.train_step(buffer)
        >>> trainer.update_action_nets()
    """
    
    value_nets: Dict[str, nn.Module]
    action_nets: Dict[str, nn.Module]
    config: TrainingConfig = field(default_factory=TrainingConfig)
    
    _optimizers: Dict[str, Optimizer] = field(default_factory=dict)
    _criterion: nn.Module = field(default=None)
    _best_score: float = field(default=0.0)
    _scores: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize optimizers and loss function."""
        device = torch.device(self.config.device)
        
        # Move value nets to device
        for net in self.value_nets.values():
            net.to(device)
        
        # Initialize optimizers
        self._optimizers = {
            key: torch.optim.Adam(
                net.parameters(), 
                lr=self.config.learning_rate
            )
            for key, net in self.value_nets.items()
        }
        
        # Huber loss (SmoothL1Loss)
        self._criterion = nn.SmoothL1Loss(beta=30.0).to(device)
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def train_step(
        self,
        buffer: "ExperienceBuffer",
        action_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Perform one training step on buffered data.
        
        Args:
            buffer: Buffer containing training data
            action_types: Which action types to train (default: all)
            
        Returns:
            Dictionary mapping action_type -> mean loss
        """
        action_types = action_types or ['discard', 'pick', 'koikoi']
        device = torch.device(self.config.device)
        losses = {}
        
        for key in action_types:
            self.value_nets[key].train()
            train_losses = []
            
            # Get batches from buffer
            for batch in self._get_batches(buffer, key):
                states, actions, rewards = batch
                states = states.to(device)
                rewards = rewards.to(device)
                
                # Forward pass
                q_values = self.value_nets[key](states).squeeze(1)
                
                # Compute loss
                loss = self._criterion(q_values, rewards)
                
                # Backward pass
                self._optimizers[key].zero_grad()
                loss.backward()
                self._optimizers[key].step()
                
                train_losses.append(loss.item())
            
            mean_loss = np.mean(train_losses) if train_losses else 0.0
            losses[key] = mean_loss
            print_log(
                f'{time_str()} {key} net, {len(train_losses)} steps, '
                f'loss = {mean_loss:.6f}',
                self.config.log_path
            )
        
        return losses
    
    def _get_batches(
        self, 
        buffer: Any, 
        action_type: str
    ):
        """
        Generate batches from buffer (compatible with original Buffer class).
        
        Yields (states, actions, rewards) tuples.
        """
        # Check if buffer is original Buffer class or new ExperienceBuffer
        if hasattr(buffer, 'get_batch'):
            # Original Buffer class
            from collections import namedtuple
            Transition = namedtuple('Transition', ['state', 'action', 'reward'])
            
            for batch in buffer.get_batch(action_type, self.config.batch_size):
                transitions = Transition(*zip(*batch))
                states = torch.stack(transitions.state)
                rewards = torch.FloatTensor(transitions.reward)
                yield states, transitions.action, rewards
        else:
            # New ExperienceBuffer
            while buffer.size(action_type) >= self.config.batch_size:
                states, actions, rewards, _ = buffer.sample(
                    action_type, 
                    self.config.batch_size
                )
                yield states, actions, rewards
    
    def update_action_nets(self) -> None:
        """
        Copy weights from value networks to action networks.
        
        This implements the target network update in DQN-style
        training. Called periodically to stabilize training.
        """
        for key in self.value_nets:
            self.action_nets[key].load_state_dict(
                self.value_nets[key].state_dict()
            )
        print_log(
            f'{time_str()} Action networks updated.',
            self.config.log_path
        )
    
    def save_models(self, loop: int, score: float) -> None:
        """
        Save current action networks.
        
        Args:
            loop: Current training loop number
            score: Current evaluation score
        """
        for key in self.action_nets:
            path = Path(self.config.save_dir) / f'{key}_{loop}_{int(score*100)}.pt'
            torch.save(self.action_nets[key], path)
        
        print_log(
            f'{time_str()} Models saved at loop {loop}, score {score:.3f}',
            self.config.log_path
        )
    
    def should_save(self, score: float) -> bool:
        """
        Determine if models should be saved based on score.
        
        Saves if score is best in last 20 evaluations.
        
        Args:
            score: Current evaluation score
            
        Returns:
            True if should save models
        """
        self._scores.append(score)
        recent_scores = self._scores[-20:] if len(self._scores) >= 20 else self._scores
        return score >= max(recent_scores)


@dataclass  
class Arena:
    """
    Arena for evaluating agents against each other.
    
    Runs multiple games between two agents and collects
    statistics on win rates and points.
    
    Attributes:
        agent1: First agent (usually training agent)
        agent2: Second agent (usually master/baseline)
    
    Example:
        >>> arena = Arena(training_agent, master_agent)
        >>> arena.multi_game_test(100)
        >>> print(f"Win rate: {arena.win_rate}")
    """
    
    agent1: "KoiKoiAgent"
    agent2: "KoiKoiAgent"
    
    test_win_num: List[int] = field(default_factory=lambda: [0, 0, 0])
    test_point: Dict[int, List[float]] = field(default_factory=lambda: {1: [], 2: []})
    
    def __post_init__(self):
        """Reset statistics."""
        self.test_win_num = [0, 0, 0]  # [draw, agent1_win, agent2_win]
        self.test_point = {1: [], 2: []}
    
    def single_game_test(self) -> Tuple[int, Dict[int, float]]:
        """
        Run a single test game.
        
        Returns:
            Tuple of (winner, point_dict)
        """
        from koikoi.core.game_state import KoiKoiGameState
        
        game_state = KoiKoiGameState()
        agents = {1: self.agent1, 2: self.agent2}
        
        while not game_state.game_over:
            while not game_state.round_state.round_over:
                player = game_state.round_state.turn_player
                if player in [1, 2]:
                    action = agents[player].select_action(
                        game_state,
                        action_mask=game_state.get_action_mask()
                    )
                else:
                    action = None
                game_state.round_state.step(action)
            game_state.new_round()
        
        # Determine winner
        if game_state.point[1] > game_state.point[2]:
            winner = 1
        elif game_state.point[2] > game_state.point[1]:
            winner = 2
        else:
            winner = 0  # Draw
        
        return winner, game_state.point
    
    def multi_game_test(self, n_games: int) -> None:
        """
        Run multiple test games and collect statistics.
        
        Args:
            n_games: Number of games to play
        """
        for _ in range(n_games):
            winner, points = self.single_game_test()
            self.test_win_num[winner] += 1
            self.test_point[1].append(points[1])
            self.test_point[2].append(points[2])
    
    @property
    def win_rate(self) -> float:
        """Agent1 win rate (excluding draws)."""
        total = sum(self.test_win_num)
        if total == 0:
            return 0.0
        # Score: 0.5 * draw_rate + win_rate
        return (0.5 * self.test_win_num[0] + self.test_win_num[1]) / total
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get full statistics dictionary."""
        total = sum(self.test_win_num)
        return {
            'total_games': total,
            'draws': self.test_win_num[0],
            'agent1_wins': self.test_win_num[1],
            'agent2_wins': self.test_win_num[2],
            'agent1_mean_points': np.mean(self.test_point[1]) if self.test_point[1] else 0,
            'agent2_mean_points': np.mean(self.test_point[2]) if self.test_point[2] else 0,
            'win_rate': self.win_rate,
        }


def epsilon_schedule(score: float) -> List[float]:
    """
    Compute exploration rates based on current score.
    
    Decreases exploration as performance improves.
    
    Args:
        score: Current win rate score
        
    Returns:
        List of epsilon values [discard, pick_discard, pick_draw, koikoi]
    """
    thresholds = [
        (0.10, 0.25),
        (0.20, 0.20),
        (0.30, 0.15),
        (0.40, 0.125),
        (0.50, 0.10),
        (0.55, 0.075),
        (1.00, 0.05),
    ]
    
    for threshold, epsilon in thresholds:
        if score < threshold:
            return [epsilon] * 4
    
    return [0.05] * 4
