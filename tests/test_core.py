"""
Tests for the core game module.
"""

import pytest


class TestCard:
    """Tests for card representation."""
    
    def test_card_creation(self):
        """Test Card class creation and properties."""
        from koikoi.core.card import Card, CardCategory
        
        # Create a card
        crane = Card(1, 1)
        
        assert crane.suit == 1
        assert crane.rank == 1
        assert crane.is_bright
        assert crane.category == CardCategory.BRIGHT
        
    def test_card_from_index(self):
        """Test Card.from_index() class method."""
        from koikoi.core.card import Card
        
        # First card
        card0 = Card.from_index(0)
        assert card0.suit == 1
        assert card0.rank == 1
        
        # Last card
        card47 = Card.from_index(47)
        assert card47.suit == 12
        assert card47.rank == 4
        
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
        
    def test_card_sets(self):
        """Test CardSets class constants."""
        from koikoi.core.card import CardSets
        
        # Bright cards (光札)
        assert len(CardSets.BRIGHT) == 5
        assert (1, 1) in CardSets.BRIGHT  # Pine crane
        assert (3, 1) in CardSets.BRIGHT  # Cherry blossom curtain
        
        # Seed cards (タネ札)
        assert len(CardSets.SEED) == 9
        
        # Ribbon cards (短冊)
        assert len(CardSets.RIBBON) == 10


class TestYaku:
    """Tests for yaku calculation."""
    
    def test_yaku_type_enum(self):
        """Test YakuType enum values."""
        from koikoi.core.yaku import YakuType
        
        assert YakuType.FIVE_LIGHTS == 1
        assert YakuType.THREE_LIGHTS == 4
        assert YakuType.RED_RIBBONS == 12
        
    def test_yaku_dataclass(self):
        """Test Yaku dataclass creation."""
        from koikoi.core.yaku import Yaku, YakuType
        
        yaku = Yaku(
            yaku_type=YakuType.THREE_LIGHTS,
            name='Three Lights',
            name_jp='三光',
            base_points=5
        )
        
        assert yaku.base_points == 5
        assert '三光' in str(yaku)


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
        
        # Total cards: 2*8 (hands) + 8 (field) + remaining deck = 48
        # The deck is internal, just verify total card count is correct
        total_cards = len(state.hand[1]) + len(state.hand[2]) + len(field_cards)
        assert total_cards == 24  # 8 + 8 + 8, remaining 24 in deck
        
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
        
        # Feature tensor should have shape (300, 48) based on NET_PARAMETERS
        assert feature.shape[1] == 48
        assert feature.shape[0] == 300  # n_input from NET_PARAMETERS
