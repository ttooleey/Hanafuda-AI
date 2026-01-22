"""
Constants for Koi-Koi game.

This module defines all magic numbers and configuration values used throughout
the game logic, making it easier to understand and modify game parameters.
"""

from enum import IntEnum
from typing import Final


# =============================================================================
# Card Dimensions
# =============================================================================

TOTAL_CARDS: Final[int] = 48
"""Total number of cards in a Hanafuda deck."""

SUITS_COUNT: Final[int] = 12
"""Number of suits (months) in the deck."""

CARDS_PER_SUIT: Final[int] = 4
"""Number of cards per suit."""


# =============================================================================
# Game Structure
# =============================================================================

MAX_TURNS: Final[int] = 16
"""Maximum turns per round (8 per player)."""

TURNS_PER_PLAYER: Final[int] = 8
"""Number of turns each player gets per round."""

DEFAULT_ROUNDS: Final[int] = 8
"""Default number of rounds in a game."""

DEFAULT_INITIAL_POINTS: Final[int] = 30
"""Default starting points for each player."""

BANKRUPTCY_THRESHOLD: Final[int] = 0
"""Point threshold at which a player loses."""


# =============================================================================
# Hand and Field Sizes
# =============================================================================

INITIAL_HAND_SIZE: Final[int] = 8
"""Number of cards dealt to each player initially."""

INITIAL_FIELD_SIZE: Final[int] = 8
"""Number of cards placed on the field initially."""

MAX_FIELD_SLOTS: Final[int] = 18
"""Maximum field slots (8 initial + up to 10 added cards)."""


# =============================================================================
# Neural Network Feature Dimensions
# =============================================================================

FEATURE_INPUT_DIM: Final[int] = 300
"""Input dimension for feature encoding."""

FEATURE_EMBEDDING_DIM: Final[int] = 256
"""Embedding dimension in encoder block."""

FEATURE_FEEDFORWARD_DIM: Final[int] = 512
"""Feedforward dimension in encoder block."""

ATTENTION_HEADS: Final[int] = 4
"""Number of attention heads in transformer."""

ENCODER_LAYERS: Final[int] = 2
"""Number of transformer encoder layers."""


# =============================================================================
# Enums
# =============================================================================

class PlayerID(IntEnum):
    """
    Player identifiers.
    
    Attributes:
        NONE: No player (used for unassigned or draw states)
        PLAYER_1: First player
        PLAYER_2: Second player
    """
    NONE = 0
    PLAYER_1 = 1
    PLAYER_2 = 2
    
    @property
    def opponent(self) -> "PlayerID":
        """Return the opponent's player ID."""
        if self == PlayerID.PLAYER_1:
            return PlayerID.PLAYER_2
        elif self == PlayerID.PLAYER_2:
            return PlayerID.PLAYER_1
        return PlayerID.NONE
    
    def __str__(self) -> str:
        if self == PlayerID.PLAYER_1:
            return "Player 1"
        elif self == PlayerID.PLAYER_2:
            return "Player 2"
        return "None"


class Month(IntEnum):
    """
    Japanese month names for card suits.
    
    Each month corresponds to a specific flower/plant motif:
    - January: Pine (松 matsu)
    - February: Plum Blossom (梅 ume)
    - March: Cherry Blossom (桜 sakura)
    - April: Wisteria (藤 fuji)
    - May: Iris (菖蒲 ayame)
    - June: Peony (牡丹 botan)
    - July: Bush Clover (萩 hagi)
    - August: Pampas Grass (薄/芒 susuki)
    - September: Chrysanthemum (菊 kiku)
    - October: Maple (紅葉 momiji)
    - November: Willow (柳 yanagi)
    - December: Paulownia (桐 kiri)
    """
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12
    
    @property
    def japanese_name(self) -> str:
        """Return the Japanese name for the month's flower."""
        names = {
            1: "松 (Matsu/Pine)",
            2: "梅 (Ume/Plum)",
            3: "桜 (Sakura/Cherry)",
            4: "藤 (Fuji/Wisteria)",
            5: "菖蒲 (Ayame/Iris)",
            6: "牡丹 (Botan/Peony)",
            7: "萩 (Hagi/Bush Clover)",
            8: "薄 (Susuki/Pampas)",
            9: "菊 (Kiku/Chrysanthemum)",
            10: "紅葉 (Momiji/Maple)",
            11: "柳 (Yanagi/Willow)",
            12: "桐 (Kiri/Paulownia)",
        }
        return names.get(self.value, "Unknown")
