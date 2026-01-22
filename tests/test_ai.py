"""
Tests for AI module.
"""

import pytest
import torch
import numpy as np


class TestModels:
    """Tests for neural network models."""
    
    def test_discard_model_forward(self):
        """Test discard model forward pass."""
        from koikoi.ai.models import DiscardModel, NET_PARAMETERS
        
        model = DiscardModel()
        
        # Create dummy input matching expected dimensions
        batch_size = 4
        n_input = NET_PARAMETERS['n_input']  # 300
        seq_len = 48
        x = torch.randn(batch_size, n_input, seq_len)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 48)
        
    def test_pick_model_forward(self):
        """Test pick model forward pass."""
        from koikoi.ai.models import PickModel, NET_PARAMETERS
        
        model = PickModel()
        
        batch_size = 4
        n_input = NET_PARAMETERS['n_input']
        x = torch.randn(batch_size, n_input, 48)
        
        output = model(x)
        
        assert output.shape == (batch_size, 48)
        
    def test_koikoi_model_forward(self):
        """Test koi-koi model forward pass."""
        from koikoi.ai.models import KoiKoiModel, NET_PARAMETERS
        
        model = KoiKoiModel()
        
        batch_size = 4
        n_input = NET_PARAMETERS['n_input']
        # KoiKoi model expects 50 columns (2 decision + 48 cards)
        x = torch.randn(batch_size, n_input, 50)
        
        output = model(x)
        
        # Output should be 2 (stop vs continue)
        assert output.shape == (batch_size, 2)


class TestStrategies:
    """Tests for action selection strategies."""
    
    def test_random_strategy(self):
        """Test random strategy returns valid actions."""
        from koikoi.ai.strategies import RandomStrategy
        from koikoi.core.game_state import KoiKoiGameState
        
        strategy = RandomStrategy()
        game = KoiKoiGameState()
        
        # Create a dummy mask (all valid)
        mask = np.ones(48)
        
        action = strategy.select_action(game, mask)
        
        # Action should be a card [suit, rank]
        assert isinstance(action, list)
        assert len(action) == 2
        assert 1 <= action[0] <= 12  # Valid suit
        assert 1 <= action[1] <= 4   # Valid rank


class TestExperienceBuffer:
    """Tests for experience replay buffer."""
    
    def test_buffer_push_and_sample(self):
        """Test pushing and sampling from buffer."""
        from koikoi.training.buffer import ExperienceBuffer, Experience
        
        buffer = ExperienceBuffer(capacity=100)
        
        # Push some experiences
        for i in range(50):
            exp = Experience(
                state=np.random.randn(300, 48),
                action=[1, 1],
                reward=1.0,
                action_type='discard',
                action_mask=np.ones(48),
            )
            buffer.push(exp)
        
        # Check size
        assert buffer.size('discard') == 50
        
        # Sample batch - returns tuple of tensors, check first element (states)
        batch = buffer.sample('discard', batch_size=10)
        # batch is a tuple (states, actions, rewards, masks)
        assert batch[0].shape[0] == 10  # 10 samples
        
    def test_buffer_ring_behavior(self):
        """Test ring buffer overflow behavior."""
        from koikoi.training.buffer import ExperienceBuffer, Experience
        
        capacity = 10
        buffer = ExperienceBuffer(capacity=capacity)
        
        # Push more than capacity
        for i in range(20):
            exp = Experience(
                state=np.random.randn(300, 48),
                action=[1, 1],
                reward=float(i),
                action_type='discard',
                action_mask=np.ones(48),
            )
            buffer.push(exp)
        
        # Size should be capped at capacity
        assert buffer.size('discard') == capacity
