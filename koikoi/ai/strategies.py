"""
Action strategies for Koi-Koi AI using Strategy Pattern.

This module defines different strategies for choosing actions,
allowing flexible composition of agent behaviors.

Design Pattern: Strategy Pattern
    - ActionStrategy: Abstract base defining the interface
    - RandomStrategy: Uniform random action selection
    - ModelBasedStrategy: Neural network-based selection
    - EpsilonGreedyStrategy: Combines random and model-based

This allows agents to have interchangeable decision-making behaviors.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from koikoi.core.constants import Action

if TYPE_CHECKING:
    from koikoi.core.game_state import KoiKoiGameState


class ActionStrategy(ABC):
    """
    Abstract base class for action selection strategies.
    
    Implementations define how an agent chooses actions given
    a game state. This follows the Strategy Pattern to allow
    interchangeable action selection behaviors.
    
    Example:
        >>> class MyStrategy(ActionStrategy):
        ...     def select_action(self, game_state, action_mask):
        ...         # Custom action selection logic
        ...         return action
    """
    
    @abstractmethod
    def select_action(
        self,
        game_state: "KoiKoiGameState",
        action_mask: np.ndarray
    ) -> Action:
        """
        Select an action for the current game state.
        
        Args:
            game_state: Current game state
            action_mask: Valid action mask (1 for valid, 0 for invalid)
            
        Returns:
            Selected action:
                - For discard/pick: [suit, rank] list
                - For koi-koi: True (continue) or False (stop)
        """
        pass


class RandomStrategy(ActionStrategy):
    """
    Strategy that selects actions uniformly at random.
    
    Useful for:
    - Exploration during training
    - Baseline comparison
    - Testing game logic
    
    Example:
        >>> strategy = RandomStrategy()
        >>> action = strategy.select_action(game_state, mask)
    """
    
    def select_action(
        self,
        game_state: "KoiKoiGameState",
        action_mask: np.ndarray
    ) -> Action:
        """Select a random valid action."""
        round_state = game_state.round_state
        phase = round_state.state
        
        if phase == 'discard':
            player = round_state.turn_player
            hand = round_state.hand[player]
            return random.choice(hand)
        
        elif phase in ('discard-pick', 'draw-pick'):
            pairing = round_state.pairing_card
            return random.choice(pairing)
        
        elif phase == 'koikoi':
            return random.choice([True, False])
        
        return None


class ModelBasedStrategy(ActionStrategy):
    """
    Strategy that uses neural network models for action selection.
    
    This strategy maintains separate models for different action types
    (discard, pick, koi-koi) and uses them to score possible actions.
    The action with the highest score (after masking) is selected.
    
    Attributes:
        models: Dictionary mapping action type to PyTorch model
        temperature: Softmax temperature for action probabilities
            Higher temperature = more random
            Lower temperature = more greedy
    
    Example:
        >>> strategy = ModelBasedStrategy(
        ...     discard_model=discard_net,
        ...     pick_model=pick_net,
        ...     koikoi_model=koikoi_net,
        ...     temperature=10.0
        ... )
        >>> action = strategy.select_action(game_state, mask)
    """
    
    # Mapping from game phase to model key
    PHASE_TO_MODEL_KEY: Dict[str, str] = {
        'discard': 'discard',
        'discard-pick': 'pick',
        'draw-pick': 'pick',
        'koikoi': 'koikoi',
    }
    
    def __init__(
        self,
        discard_model: torch.nn.Module,
        pick_model: torch.nn.Module,
        koikoi_model: torch.nn.Module,
        temperature: float = 10.0,
    ) -> None:
        """
        Initialize with trained models.
        
        Args:
            discard_model: Model for discard decisions
            pick_model: Model for pick decisions  
            koikoi_model: Model for koi-koi decisions
            temperature: Softmax temperature (higher = more random)
        """
        self.models = {
            'discard': discard_model,
            'pick': pick_model,
            'koikoi': koikoi_model,
        }
        self.temperature = temperature
        
        # Set all models to evaluation mode
        for model in self.models.values():
            model.eval()
        
        # Precompute action mappings
        self._card_actions: List[List[int]] = [
            [suit + 1, rank + 1]
            for suit in range(12)
            for rank in range(4)
        ]
        self._koikoi_actions = (False, True)
    
    def select_action(
        self,
        game_state: "KoiKoiGameState",
        action_mask: np.ndarray
    ) -> Action:
        """
        Select action using model predictions.
        
        The model outputs are passed through softmax with temperature,
        then masked by valid actions, and the highest-scoring action
        is selected.
        """
        phase = game_state.round_state.state
        round_state = game_state.round_state
        
        if phase not in self.PHASE_TO_MODEL_KEY:
            return None
        
        model_key = self.PHASE_TO_MODEL_KEY[phase]
        model = self.models[model_key]
        
        # Get features and run inference
        feature = game_state.feature_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(feature).squeeze(0).numpy()
        
        # Build valid action mask based on game state
        if phase == 'koikoi':
            valid_mask = np.ones(2)
        elif phase == 'discard':
            # Only cards in hand are valid
            valid_mask = np.zeros(48)
            player = round_state.turn_player
            for card in round_state.hand[player]:
                idx = 4 * (card[0] - 1) + (card[1] - 1)
                valid_mask[idx] = 1
        elif phase in ('discard-pick', 'draw-pick'):
            # Only pairing cards are valid
            valid_mask = np.zeros(48)
            for card in round_state.pairing_card:
                idx = 4 * (card[0] - 1) + (card[1] - 1)
                valid_mask[idx] = 1
        else:
            valid_mask = action_mask
        
        # Apply softmax with temperature and mask invalid actions
        scores = np.exp(output / self.temperature) * valid_mask
        
        # Handle case where all scores are 0
        if scores.sum() == 0:
            # Fallback to random valid action
            valid_indices = np.where(valid_mask > 0)[0]
            if len(valid_indices) == 0:
                return None
            best_index = int(np.random.choice(valid_indices))
        else:
            best_index = int(scores.argmax())
        
        # Map index to action
        if phase == 'koikoi':
            return self._koikoi_actions[best_index]
        else:
            return self._card_actions[best_index]


class EpsilonGreedyStrategy(ActionStrategy):
    """
    Strategy combining model-based and random action selection.
    
    With probability epsilon, a random action is chosen (exploration).
    Otherwise, the model-based action is chosen (exploitation).
    
    This is useful for reinforcement learning where we need to
    balance exploration and exploitation.
    
    The epsilon can vary by game phase, allowing different
    exploration rates for different decision types.
    
    Attributes:
        model_strategy: The underlying model-based strategy
        random_strategy: Random strategy for exploration
        epsilon: Dictionary mapping phase -> exploration probability
    
    Example:
        >>> strategy = EpsilonGreedyStrategy(
        ...     model_strategy=model_strat,
        ...     epsilon={'discard': 0.1, 'pick': 0.1, 'koikoi': 0.1}
        ... )
    """
    
    def __init__(
        self,
        model_strategy: ModelBasedStrategy,
        epsilon: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize with model strategy and exploration rates.
        
        Args:
            model_strategy: Strategy to use for exploitation
            epsilon: Dict mapping phase -> exploration probability
                If None, defaults to 0 (no exploration)
        """
        self.model_strategy = model_strategy
        self.random_strategy = RandomStrategy()
        
        # Default: no exploration
        self.epsilon = epsilon or {
            'discard': 0.0,
            'discard-pick': 0.0,
            'draw-pick': 0.0,
            'koikoi': 0.0,
        }
    
    def select_action(
        self,
        game_state: "KoiKoiGameState",
        action_mask: np.ndarray
    ) -> Action:
        """
        Select action with epsilon-greedy exploration.
        
        With probability epsilon[phase], select random action.
        Otherwise, use model-based selection.
        """
        phase = game_state.round_state.state
        eps = self.epsilon.get(phase, 0.0)
        
        if random.random() < eps:
            return self.random_strategy.select_action(game_state, action_mask)
        else:
            return self.model_strategy.select_action(game_state, action_mask)
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        Set uniform epsilon for all phases.
        
        Args:
            epsilon: Exploration probability (0 to 1)
        """
        for phase in self.epsilon:
            self.epsilon[phase] = epsilon
    
    def set_epsilon_by_phase(self, phase: str, epsilon: float) -> None:
        """
        Set epsilon for a specific phase.
        
        Args:
            phase: Game phase name
            epsilon: Exploration probability (0 to 1)
        """
        if phase in self.epsilon:
            self.epsilon[phase] = epsilon
