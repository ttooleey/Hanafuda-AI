"""
Tests for the core game module.
"""

import pytest


class TestCard:
    """Tests for card representation."""
    
    def test_card_encoding(self):
        """Test multi-hot card encoding."""
        from koikoi.core.card import CardEncoder
        
        encoder = CardEncoder()
        
        # Test single card encoding
        encoding = encoder.encode_cards([[1, 1]])  # First card
        assert len(encoding) == 48  # Total cards
        assert sum(encoding) == 1  # One card encoded
        
    def test_card_classification(self):
        """Test card category classification."""
        from koikoi.core.card import CardCategory, CardSets
        
        # Bright cards (光札)
        bright_cards = CardSets.BRIGHT_CARDS
        assert len(bright_cards) == 5
        assert [1, 1] in bright_cards  # Pine crane
        assert [3, 1] in bright_cards  # Cherry blossom curtain
        
    def test_card_to_index(self):
        """Test card to index conversion."""
        from koikoi.utils.helpers import action_to_index, index_to_action
        
        # First card
        assert action_to_index([1, 1]) == 0
        
        # Last card
        assert action_to_index([12, 4]) == 47
        
        # Reverse conversion
        assert index_to_action(0, 'card') == [1, 1]
        assert index_to_action(47, 'card') == [12, 4]
        
        # Koi-koi actions
        assert action_to_index(True) == 1
        assert action_to_index(False) == 0
        assert index_to_action(1, 'koikoi') == True
        assert index_to_action(0, 'koikoi') == False


class TestYaku:
    """Tests for yaku calculation."""
    
    def test_sanko(self):
        """Test 三光 (Three Brights) yaku."""
        from koikoi.core.yaku import YakuCalculator
        
        calculator = YakuCalculator()
        
        # Three brights without rain man
        pile = [[1, 1], [3, 1], [8, 1]]  # Pine, Cherry, Moon
        yakus = calculator.calculate(pile)
        
        yaku_names = [y.name for y in yakus]
        assert '三光' in yaku_names
        
    def test_akatan(self):
        """Test 赤短 (Red Poetry Ribbons) yaku."""
        from koikoi.core.yaku import YakuCalculator
        
        calculator = YakuCalculator()
        
        # Three poetry ribbons
        pile = [[1, 2], [2, 2], [3, 2]]  # Pine, Plum, Cherry ribbons
        yakus = calculator.calculate(pile)
        
        yaku_names = [y.name for y in yakus]
        assert '赤短' in yaku_names


class TestRoundState:
    """Tests for round state management."""
    
    def test_initial_state(self):
        """Test initial round state."""
        from koikoi.core.round_state import KoiKoiRoundState
        
        state = KoiKoiRoundState()
        
        # Initial hands should have 8 cards each
        assert len(state.hand[1]) == 8
        assert len(state.hand[2]) == 8
        
        # Field should have 8 cards
        field_cards = [c for c in state.field_slot if c != [0, 0]]
        assert len(field_cards) == 8
        
        # Deck should have 24 cards
        assert len(state.deck) == 24
        
    def test_state_transitions(self):
        """Test state machine transitions."""
        from koikoi.core.round_state import KoiKoiRoundState
        
        state = KoiKoiRoundState()
        
        # Initial state should be 'discard'
        assert state.state == 'discard'
        
        # Discard should transition to 'discard-pick' or 'draw'
        card = state.hand[state.turn_player][0]
        state.discard(card)
        assert state.state in ['discard-pick', 'draw']


class TestGameState:
    """Tests for multi-round game management."""
    
    def test_initial_game_state(self):
        """Test initial game state."""
        from koikoi.core.game_state import KoiKoiGameState
        
        game = KoiKoiGameState()
        
        assert game.round == 1
        assert game.round_total == 8
        assert game.point[1] == 30
        assert game.point[2] == 30
        assert not game.game_over
        
    def test_feature_tensor(self):
        """Test feature tensor generation."""
        from koikoi.core.game_state import KoiKoiGameState
        
        game = KoiKoiGameState()
        
        feature = game.feature_tensor
        
        # Feature tensor should have shape (206, 48)
        assert feature.shape == (206, 48)
