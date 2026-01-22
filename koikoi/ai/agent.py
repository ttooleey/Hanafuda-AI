"""
Koi-Koi AI Agent implementation.

This module provides the KoiKoiAgent class that encapsulates
all AI decision-making logic, combining models and strategies
into a cohesive interface.

The agent follows the Facade Pattern, providing a simple
interface to the complex underlying strategy and model systems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import torch

from koikoi.ai.models import DiscardModel, PickModel, KoiKoiModel
from koikoi.ai.strategies import (
    ActionStrategy,
    EpsilonGreedyStrategy,
    ModelBasedStrategy,
    RandomStrategy,
)

if TYPE_CHECKING:
    from koikoi.core.game_state import KoiKoiGameState


class KoiKoiAgent:
    """
    Koi-Koi AI agent that selects actions given game states.
    
    This class acts as a facade for the strategy system,
    providing a clean interface for game integration.
    
    The agent can operate in different modes:
    - Random: Pure random action selection
    - Model: Pure model-based selection (greedy)
    - Training: Epsilon-greedy for exploration/exploitation
    
    Attributes:
        strategy: The action selection strategy being used
        models: Dictionary of loaded neural network models
        device: Device for model inference (cpu/cuda)
    
    Example:
        >>> # Load pretrained agent
        >>> agent = KoiKoiAgent.load_pretrained("model_agent/")
        >>> 
        >>> # Get action for current state
        >>> action = agent.select_action(game_state, action_mask)
        >>> 
        >>> # Set exploration rate for training
        >>> agent.set_epsilon(0.1)
    """
    
    DEFAULT_MODEL_PARAMS = {
        'input_dim': 206,
        'output_dim_discard': 48,
        'output_dim_pick': 48,
        'output_dim_koikoi': 2,
    }
    
    def __init__(
        self,
        strategy: ActionStrategy,
        models: Optional[Dict[str, torch.nn.Module]] = None,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize agent with a strategy.
        
        Args:
            strategy: Action selection strategy to use
            models: Dictionary of models (for reference)
            device: Device for inference
        """
        self.strategy = strategy
        self.models = models or {}
        self.device = device
    
    @classmethod
    def create_random(cls) -> "KoiKoiAgent":
        """
        Create an agent that plays randomly.
        
        Returns:
            KoiKoiAgent with RandomStrategy
        
        Example:
            >>> agent = KoiKoiAgent.create_random()
        """
        return cls(strategy=RandomStrategy())
    
    @classmethod
    def load_pretrained(
        cls,
        model_dir: Union[str, Path],
        temperature: float = 10.0,
        device: str = 'cpu',
    ) -> "KoiKoiAgent":
        """
        Load a pretrained agent from model directory.
        
        Expects the directory to contain:
        - discard.model: Discard decision model
        - pick.model: Pick decision model  
        - koikoi.model: Koi-koi decision model
        
        Args:
            model_dir: Path to directory containing model files
            temperature: Softmax temperature for action selection
            device: Device for model inference
            
        Returns:
            KoiKoiAgent with loaded models
            
        Raises:
            FileNotFoundError: If model files don't exist
        
        Example:
            >>> agent = KoiKoiAgent.load_pretrained("model_agent/")
        """
        model_path = Path(model_dir)
        
        # Create model instances
        discard_model = DiscardModel(
            cls.DEFAULT_MODEL_PARAMS['input_dim'],
            cls.DEFAULT_MODEL_PARAMS['output_dim_discard'],
        )
        pick_model = PickModel(
            cls.DEFAULT_MODEL_PARAMS['input_dim'],
            cls.DEFAULT_MODEL_PARAMS['output_dim_pick'],
        )
        koikoi_model = KoiKoiModel(
            cls.DEFAULT_MODEL_PARAMS['input_dim'],
            cls.DEFAULT_MODEL_PARAMS['output_dim_koikoi'],
        )
        
        # Load weights
        discard_model.load_state_dict(
            torch.load(model_path / "discard.model", map_location=device)
        )
        pick_model.load_state_dict(
            torch.load(model_path / "pick.model", map_location=device)
        )
        koikoi_model.load_state_dict(
            torch.load(model_path / "koikoi.model", map_location=device)
        )
        
        # Move to device
        discard_model.to(device)
        pick_model.to(device)
        koikoi_model.to(device)
        
        # Create model-based strategy
        strategy = ModelBasedStrategy(
            discard_model=discard_model,
            pick_model=pick_model,
            koikoi_model=koikoi_model,
            temperature=temperature,
        )
        
        models = {
            'discard': discard_model,
            'pick': pick_model,
            'koikoi': koikoi_model,
        }
        
        return cls(strategy=strategy, models=models, device=device)
    
    @classmethod
    def create_for_training(
        cls,
        discard_model: torch.nn.Module,
        pick_model: torch.nn.Module,
        koikoi_model: torch.nn.Module,
        epsilon: float = 0.1,
        temperature: float = 10.0,
        device: str = 'cpu',
    ) -> "KoiKoiAgent":
        """
        Create an agent for reinforcement learning training.
        
        Uses epsilon-greedy strategy to balance exploration
        and exploitation during training.
        
        Args:
            discard_model: Discard decision model
            pick_model: Pick decision model
            koikoi_model: Koi-koi decision model
            epsilon: Exploration probability
            temperature: Softmax temperature
            device: Device for inference
            
        Returns:
            KoiKoiAgent with EpsilonGreedyStrategy
        
        Example:
            >>> agent = KoiKoiAgent.create_for_training(
            ...     discard_model, pick_model, koikoi_model,
            ...     epsilon=0.1
            ... )
        """
        model_strategy = ModelBasedStrategy(
            discard_model=discard_model,
            pick_model=pick_model,
            koikoi_model=koikoi_model,
            temperature=temperature,
        )
        
        epsilon_strategy = EpsilonGreedyStrategy(
            model_strategy=model_strategy,
            epsilon={
                'discard': epsilon,
                'discard-pick': epsilon,
                'draw-pick': epsilon,
                'koikoi': epsilon,
            },
        )
        
        models = {
            'discard': discard_model,
            'pick': pick_model,
            'koikoi': koikoi_model,
        }
        
        return cls(strategy=epsilon_strategy, models=models, device=device)
    
    def select_action(
        self,
        game_state: "KoiKoiGameState",
        action_mask: Any,
    ) -> Any:
        """
        Select an action for the current game state.
        
        Delegates to the underlying strategy for action selection.
        
        Args:
            game_state: Current game state
            action_mask: Valid action mask
            
        Returns:
            Selected action (format depends on game phase)
        """
        return self.strategy.select_action(game_state, action_mask)
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        Set exploration rate (for training agents).
        
        Only works if using EpsilonGreedyStrategy.
        
        Args:
            epsilon: Exploration probability (0 to 1)
            
        Raises:
            TypeError: If strategy doesn't support epsilon
        """
        if isinstance(self.strategy, EpsilonGreedyStrategy):
            self.strategy.set_epsilon(epsilon)
        else:
            raise TypeError(
                f"Cannot set epsilon on {type(self.strategy).__name__}. "
                "Use EpsilonGreedyStrategy for epsilon support."
            )
    
    def set_eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        for model in self.models.values():
            model.eval()
    
    def set_train_mode(self) -> None:
        """Set all models to training mode."""
        for model in self.models.values():
            model.train()
    
    def save_models(self, model_dir: Union[str, Path]) -> None:
        """
        Save all models to a directory.
        
        Args:
            model_dir: Directory to save models to
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        model_files = {
            'discard': 'discard.model',
            'pick': 'pick.model',
            'koikoi': 'koikoi.model',
        }
        
        for key, filename in model_files.items():
            if key in self.models:
                torch.save(
                    self.models[key].state_dict(),
                    model_path / filename
                )
