"""
Tests for AI module.
"""

import pytest
import torch


class TestModels:
    """Tests for neural network models."""
    
    def test_discard_model_forward(self):
        """Test discard model forward pass."""
        from koikoi.ai.models import DiscardModel
        
        model = DiscardModel(input_dim=206, output_dim=48)
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 206, 48)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 48)
        
    def test_pick_model_forward(self):
        """Test pick model forward pass."""
        from koikoi.ai.models import PickModel
        
        model = PickModel(input_dim=206, output_dim=48)
        
        batch_size = 4
        x = torch.randn(batch_size, 206, 48)
        
        output = model(x)
        
        assert output.shape == (batch_size, 48)
        
    def test_koikoi_model_forward(self):
        """Test koi-koi model forward pass."""
        from koikoi.ai.models import KoiKoiModel
        
        model = KoiKoiModel(input_dim=206, output_dim=2)
        
        batch_size = 4
        x = torch.randn(batch_size, 206, 48)
        
        output = model(x)
        
        assert output.shape == (batch_size, 2)


class TestStrategies:
    """Tests for action selection strategies."""
    
    def test_random_strategy(self):
        """Test random strategy returns valid actions."""
        from koikoi.ai.strategies import RandomStrategy
        from koikoi.core.game_state import KoiKoiGameState
        import numpy as np
        
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
        import numpy as np
        
        buffer = ExperienceBuffer(capacity=100)
        
        # Push some experiences
        for i in range(50):
            exp = Experience(
                state=np.random.randn(206, 48),
                action=[1, 1],
                reward=1.0,
                action_type='discard',
                action_mask=np.ones(48),
            )
            buffer.push(exp)
        
        # Check size
        assert buffer.size('discard') == 50
        
        # Sample batch
        states, actions, rewards, masks = buffer.sample('discard', batch_size=10)
        
        assert states.shape[0] == 10
        assert len(actions) == 10
        assert rewards.shape[0] == 10
        assert masks.shape[0] == 10
        
    def test_buffer_ring_behavior(self):
        """Test ring buffer overwrites oldest entries."""
        from koikoi.training.buffer import ExperienceBuffer, Experience
        import numpy as np
        
        buffer = ExperienceBuffer(capacity=10)
        
        # Push more than capacity
        for i in range(15):
            exp = Experience(
                state=np.random.randn(206, 48),
                action=[1, 1],
                reward=float(i),  # Use index as reward for identification
                action_type='discard',
                action_mask=np.ones(48),
            )
            buffer.push(exp)
        
        # Should still have only capacity items
        assert buffer.size('discard') == 10
