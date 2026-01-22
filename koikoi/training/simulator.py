"""
Trace simulator for self-play game simulation.

This module provides functionality to simulate complete games
using AI agents and collect training data (traces).

The TraceSimulator class is the core component for reinforcement
learning data collection in the Koi-Koi project.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import torch

from koikoi.training.buffer import Experience

if TYPE_CHECKING:
    from koikoi.ai.agent import KoiKoiAgent


# Named tuple for storing trace information
TraceSlot = namedtuple('TraceSlot', ['key', 'state', 'action'])

# Map game states to model types
STATE_TO_ACTION_TYPE: Dict[str, str] = {
    'discard': 'discard',
    'discard-pick': 'pick',
    'draw-pick': 'pick',
    'koikoi': 'koikoi',
}


def action_to_index(action: Any) -> Optional[int]:
    """
    Convert action to integer index.
    
    Args:
        action: Action in game format
            - [suit, rank] for card actions
            - True/False for koi-koi decisions
            - None for no action
            
    Returns:
        Integer index:
            - 0-47 for card actions (suit-1)*4 + (rank-1)
            - 0 or 1 for koi-koi decisions
            - None if input is None
    """
    if action in [False, True]:
        return int(action)
    elif action is not None:
        return 4 * (action[0] - 1) + (action[1] - 1)
    return None


def adjust_card_order(feature: torch.Tensor, index: int) -> torch.Tensor:
    """
    Adjust feature tensor to put selected card first.
    
    This is used to create augmented training data by
    rotating the card order in the feature representation.
    
    Args:
        feature: Feature tensor of shape (feature_dim, num_cards)
        index: Index of the selected card
        
    Returns:
        Reordered feature tensor with selected card first
    """
    num_cards = feature.size(1)
    ind_list = [index] + [ii for ii in range(num_cards) if ii != index]
    return feature[:, ind_list]


@dataclass
class TraceSimulator:
    """
    Simulator for collecting game traces through self-play.
    
    The simulator plays complete games using the provided agent(s)
    and collects state-action-reward tuples for training.
    
    Attributes:
        agent: Agent to use for both players (self-play)
        record_states: Game states to record transitions from
        discount: Discount factor for reward propagation
        reward_function: Function to compute rewards
        
    Design Notes:
        - Uses Monte-Carlo style reward propagation (all steps
          in a round receive the final reward with discounting)
        - Supports parallel game simulation via pickle
        - Collects traces by action type for balanced training
    
    Example:
        >>> simulator = TraceSimulator(agent=play_agent)
        >>> traces = simulator.random_make_games(100)
        >>> print(len(traces['discard']))  # Number of discard decisions
    """
    
    agent: "KoiKoiAgent"
    record_states: List[str] = field(default_factory=lambda: [
        'discard', 'discard-pick', 'draw-pick', 'koikoi'
    ])
    discount: float = 1.0
    reward_function: str = 'point'  # 'point' or 'wp' (win probability)
    
    # Win probability matrix for wp reward (loaded externally if needed)
    win_prob_mat: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize the trace buffer."""
        self._buffer: Dict[str, List] = {
            'discard': [],
            'pick': [],
            'koikoi': [],
        }
    
    def _compute_reward_point(self, game_state: Any, player: int) -> float:
        """
        Compute reward based on round points scored.
        
        Simple reward: just the points scored this round.
        
        Args:
            game_state: Current game state
            player: Player ID
            
        Returns:
            Round points as float
        """
        round_point = game_state.round_state.round_point[player]
        return float(round_point)
    
    def _compute_reward_wp(self, game_state: Any, player: int) -> float:
        """
        Compute reward based on win probability.
        
        Uses precomputed win probability matrix to estimate
        expected win probability from current game state.
        
        Args:
            game_state: Current game state
            player: Player ID
            
        Returns:
            Scaled win probability (0-10 range)
        """
        if self.win_prob_mat is None:
            # Fall back to point-based reward
            return self._compute_reward_point(game_state, player)
        
        round_num = game_state.round + 1
        point = (game_state.point[player] + 
                game_state.round_state.round_point[player])
        is_dealer = int(game_state.round_state.winner == player)
        
        if round_num <= 8 and (0 < point < 60):
            win_prob = self.win_prob_mat[is_dealer, round_num, point]
        else:
            win_prob = 0.5 if point == 30 else float(point > 30)
        
        return win_prob * 10.0
    
    def _get_reward(self, game_state: Any, player: int) -> float:
        """
        Get reward based on configured reward function.
        
        Args:
            game_state: Current game state
            player: Player ID
            
        Returns:
            Computed reward
        """
        if self.reward_function == 'wp':
            return self._compute_reward_wp(game_state, player)
        else:
            return self._compute_reward_point(game_state, player)
    
    def random_make_games(self, n_games: int) -> Dict[str, List]:
        """
        Simulate multiple games and collect traces.
        
        Args:
            n_games: Number of games to simulate
            
        Returns:
            Dictionary mapping action type -> list of transitions
        """
        # Reset buffer
        self._buffer = {
            'discard': [],
            'pick': [],
            'koikoi': [],
        }
        
        for _ in range(n_games):
            self._make_game_trace()
        
        return self._buffer
    
    def _make_game_trace(self) -> None:
        """
        Simulate one complete game and record traces.
        
        Records state-action pairs during play, then propagates
        rewards back through the trace using Monte-Carlo style
        updates with discounting.
        """
        # Import here to avoid circular dependency
        from koikoi.core.game_state import KoiKoiGameState
        
        game_state = KoiKoiGameState()
        
        while not game_state.game_over:
            # Play one round and collect trace
            trace = {1: [], 2: []}
            
            while not game_state.round_state.round_over:
                player = game_state.round_state.turn_player
                state = game_state.round_state.state
                
                # Get action from agent
                action = self.agent.select_action(
                    game_state,
                    action_mask=game_state.get_action_mask()
                )
                
                # Record if this is a state we're tracking
                if (player in [1, 2] and 
                    state in self.record_states and 
                    action is not None):
                    
                    trace[player].append(TraceSlot(
                        key=STATE_TO_ACTION_TYPE[state],
                        state=game_state.feature_tensor.clone(),
                        action=action_to_index(action),
                    ))
                
                # Take action
                game_state.round_state.step(action)
            
            # Compute rewards after round ends
            rewards = {
                1: self._get_reward(game_state, 1),
                2: self._get_reward(game_state, 2),
            }
            
            # Record transitions with discounted rewards
            for player in [1, 2]:
                for rev_step, slot in enumerate(reversed(trace[player])):
                    action_idx = slot.action
                    discounted_reward = rewards[player] * (self.discount ** rev_step)
                    
                    # Adjust card order for data augmentation
                    adjusted_state = adjust_card_order(
                        slot.state.clone(), 
                        action_idx
                    )
                    
                    # Create named tuple matching original format
                    Transition = namedtuple(
                        'Transition', ['state', 'action', 'reward']
                    )
                    self._buffer[slot.key].append(Transition(
                        state=adjusted_state,
                        action=action_idx,
                        reward=discounted_reward,
                    ))
            
            # Start next round
            game_state.new_round()


def create_parallel_sampler(agent: "KoiKoiAgent", n_games: int) -> Dict[str, List]:
    """
    Function for parallel game simulation.
    
    This function is designed to be called via multiprocessing.Pool
    for parallel data collection.
    
    Args:
        agent: Agent to use for self-play
        n_games: Number of games to simulate
        
    Returns:
        Dictionary of collected traces
        
    Example:
        >>> pool = multiprocessing.Pool(cpu_count)
        >>> for _ in range(cpu_count):
        ...     pool.apply_async(
        ...         create_parallel_sampler,
        ...         args=(play_agent, n_core_games),
        ...         callback=buffer.extend
        ...     )
    """
    simulator = TraceSimulator(agent=agent)
    return simulator.random_make_games(n_games)
