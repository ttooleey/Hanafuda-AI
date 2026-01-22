"""Constants for Koi-Koi game."""

from enum import IntEnum
from typing import Final, List, Union


# Type Aliases
Card = List[int]  # [suit, rank]
CardAction = List[int]  # [suit, rank]
KoiKoiAction = bool  # True=continue, False=stop
Action = Union[CardAction, KoiKoiAction, None]
ActionType = str  # 'discard', 'pick', 'koikoi'


# Card Dimensions
TOTAL_CARDS: Final[int] = 48
SUITS_COUNT: Final[int] = 12
CARDS_PER_SUIT: Final[int] = 4


# Game Structure
MAX_TURNS: Final[int] = 16
TURNS_PER_PLAYER: Final[int] = 8
DEFAULT_ROUNDS: Final[int] = 8
DEFAULT_INITIAL_POINTS: Final[int] = 30
BANKRUPTCY_THRESHOLD: Final[int] = 0


# Hand and Field Sizes
INITIAL_HAND_SIZE: Final[int] = 8
INITIAL_FIELD_SIZE: Final[int] = 8
MAX_FIELD_SLOTS: Final[int] = 18


# Neural Network Dimensions
FEATURE_INPUT_DIM: Final[int] = 300
FEATURE_EMBEDDING_DIM: Final[int] = 256
FEATURE_FEEDFORWARD_DIM: Final[int] = 512
ATTENTION_HEADS: Final[int] = 4
ENCODER_LAYERS: Final[int] = 2


# Enums

class PlayerID(IntEnum):
    """Player identifiers."""
    NONE = 0
    PLAYER_1 = 1
    PLAYER_2 = 2
    
    @property
    def opponent(self) -> "PlayerID":
        if self == PlayerID.PLAYER_1:
            return PlayerID.PLAYER_2
        elif self == PlayerID.PLAYER_2:
            return PlayerID.PLAYER_1
        return PlayerID.NONE
    
    def __str__(self) -> str:
        return ["None", "Player 1", "Player 2"][self.value]


class Month(IntEnum):
    """Japanese month names (card suits)."""
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
        names = [
            "", "松", "梅", "桜", "藤", "菖蒲", "牡丹",
            "萩", "薄", "菊", "紅葉", "柳", "桐"
        ]
        return names[self.value] if self.value < len(names) else ""
