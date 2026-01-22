"""
Card representation and classification for Koi-Koi Hanafuda.

This module defines the Card class and card categories (Bright, Seed, Ribbon, Dross),
as well as utility functions for card encoding and manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import FrozenSet, List, Tuple

from koikoi.core.constants import TOTAL_CARDS, SUITS_COUNT, CARDS_PER_SUIT


class CardCategory(Enum):
    """
    Card categories in Hanafuda deck.
    
    Each card belongs to one of four categories based on its rank/importance:
    - BRIGHT (光): 5 special cards worth the most
    - SEED (タネ): 9 animal/object cards
    - RIBBON (短冊): 10 ribbon cards
    - DROSS (カス): 24 plain cards (including Sake cup which counts as both Seed and Dross)
    """
    BRIGHT = auto()
    SEED = auto()
    RIBBON = auto()
    DROSS = auto()


@dataclass(frozen=True, order=True)
class Card:
    """
    Represents a single Hanafuda card.
    
    Hanafuda cards are identified by their suit (month, 1-12) and rank (1-4).
    The rank typically indicates the card's importance within the suit,
    with rank 1 usually being the most valuable card.
    
    Attributes:
        suit: The month/suit (1-12)
        rank: The rank within the suit (1-4)
    
    Example:
        >>> crane = Card(1, 1)  # January Crane (Bright)
        >>> print(crane.is_bright)
        True
        >>> print(crane.category)
        CardCategory.BRIGHT
    """
    suit: int
    rank: int
    
    def __post_init__(self) -> None:
        """Validate card values."""
        if not (1 <= self.suit <= SUITS_COUNT):
            raise ValueError(f"Suit must be between 1 and {SUITS_COUNT}, got {self.suit}")
        if not (1 <= self.rank <= CARDS_PER_SUIT):
            raise ValueError(f"Rank must be between 1 and {CARDS_PER_SUIT}, got {self.rank}")
    
    @classmethod
    def from_index(cls, index: int) -> "Card":
        """
        Create a card from a flat index (0-47).
        
        Args:
            index: Card index in range [0, 47]
            
        Returns:
            Card instance
            
        Example:
            >>> Card.from_index(0)
            Card(suit=1, rank=1)
            >>> Card.from_index(47)
            Card(suit=12, rank=4)
        """
        if not (0 <= index < TOTAL_CARDS):
            raise ValueError(f"Index must be between 0 and {TOTAL_CARDS - 1}")
        suit = (index // CARDS_PER_SUIT) + 1
        rank = (index % CARDS_PER_SUIT) + 1
        return cls(suit, rank)
    
    def to_index(self) -> int:
        """
        Convert card to flat index (0-47).
        
        Returns:
            Integer index representing this card
            
        Example:
            >>> Card(1, 1).to_index()
            0
            >>> Card(12, 4).to_index()
            47
        """
        return (self.suit - 1) * CARDS_PER_SUIT + (self.rank - 1)
    
    def to_list(self) -> List[int]:
        """
        Convert to [suit, rank] list format.
        
        Provided for backward compatibility with legacy code.
        """
        return [self.suit, self.rank]
    
    @classmethod
    def from_list(cls, card_list: List[int]) -> "Card":
        """
        Create from [suit, rank] list format.
        
        Args:
            card_list: List of [suit, rank]
            
        Returns:
            Card instance
        """
        return cls(card_list[0], card_list[1])
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to (suit, rank) tuple format."""
        return (self.suit, self.rank)
    
    @property
    def category(self) -> CardCategory:
        """Determine the category of this card."""
        card_tuple = self.to_tuple()
        if card_tuple in CardSets.BRIGHT:
            return CardCategory.BRIGHT
        elif card_tuple in CardSets.SEED:
            return CardCategory.SEED
        elif card_tuple in CardSets.RIBBON:
            return CardCategory.RIBBON
        return CardCategory.DROSS
    
    @property
    def is_bright(self) -> bool:
        """Check if this is a Bright card (光札)."""
        return self.to_tuple() in CardSets.BRIGHT
    
    @property
    def is_seed(self) -> bool:
        """Check if this is a Seed card (タネ札)."""
        return self.to_tuple() in CardSets.SEED
    
    @property
    def is_ribbon(self) -> bool:
        """Check if this is a Ribbon card (短冊札)."""
        return self.to_tuple() in CardSets.RIBBON
    
    @property
    def is_dross(self) -> bool:
        """Check if this is a Dross card (カス札)."""
        return self.to_tuple() in CardSets.DROSS
    
    @property
    def is_sake(self) -> bool:
        """
        Check if this is the Sake cup (菊に盃).
        
        The Sake cup is special: it counts as both Seed and Dross.
        """
        return self.to_tuple() == (9, 1)
    
    def __str__(self) -> str:
        """Return human-readable card name."""
        return f"Card({self.suit}, {self.rank})"


class CardSets:
    """
    Predefined card sets for yaku calculation.
    
    These sets define which cards belong to specific categories or
    special combinations used in yaku (winning hand) calculation.
    
    All sets use (suit, rank) tuple format for efficient lookup.
    """
    
    # =========================================================================
    # Special Individual Cards (特定札)
    # =========================================================================
    
    CRANE: FrozenSet[Tuple[int, int]] = frozenset({(1, 1)})
    """January Crane (松に鶴) - Bright"""
    
    CURTAIN: FrozenSet[Tuple[int, int]] = frozenset({(3, 1)})
    """March Curtain (桜に幕) - Bright"""
    
    MOON: FrozenSet[Tuple[int, int]] = frozenset({(8, 1)})
    """August Moon (芒に月) - Bright"""
    
    RAINMAN: FrozenSet[Tuple[int, int]] = frozenset({(11, 1)})
    """November Rainman (柳に小野道風) - Bright"""
    
    PHOENIX: FrozenSet[Tuple[int, int]] = frozenset({(12, 1)})
    """December Phoenix (桐に鳳凰) - Bright"""
    
    SAKE: FrozenSet[Tuple[int, int]] = frozenset({(9, 1)})
    """September Sake Cup (菊に盃) - counts as both Seed and Dross"""
    
    # =========================================================================
    # Category Sets (種類別)
    # =========================================================================
    
    BRIGHT: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 1),   # Crane
        (3, 1),   # Curtain
        (8, 1),   # Moon
        (11, 1),  # Rainman
        (12, 1),  # Phoenix
    })
    """All 5 Bright cards (光札)"""
    
    SEED: FrozenSet[Tuple[int, int]] = frozenset({
        (2, 1),   # Bush Warbler
        (4, 1),   # Cuckoo
        (5, 1),   # Bridge
        (6, 1),   # Butterflies
        (7, 1),   # Boar
        (8, 2),   # Geese
        (9, 1),   # Sake Cup
        (10, 1),  # Deer
        (11, 2),  # Swallow
    })
    """All 9 Seed cards (タネ札)"""
    
    RIBBON: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 2),   # Pine Ribbon (red with poetry)
        (2, 2),   # Plum Ribbon (red with poetry)
        (3, 2),   # Cherry Ribbon (red with poetry)
        (4, 2),   # Wisteria Ribbon (red plain)
        (5, 2),   # Iris Ribbon (red plain)
        (6, 2),   # Peony Ribbon (blue)
        (7, 2),   # Bush Clover Ribbon (red plain)
        (9, 2),   # Chrysanthemum Ribbon (blue)
        (10, 2),  # Maple Ribbon (blue)
        (11, 3),  # Willow Ribbon (red plain)
    })
    """All 10 Ribbon cards (短冊札)"""
    
    DROSS: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 3), (1, 4),    # January
        (2, 3), (2, 4),    # February
        (3, 3), (3, 4),    # March
        (4, 3), (4, 4),    # April
        (5, 3), (5, 4),    # May
        (6, 3), (6, 4),    # June
        (7, 3), (7, 4),    # July
        (8, 3), (8, 4),    # August
        (9, 1),            # September Sake (also Seed!)
        (9, 3), (9, 4),    # September
        (10, 3), (10, 4),  # October
        (11, 4),           # November
        (12, 2), (12, 3), (12, 4),  # December
    })
    """All Dross cards (カス札) - includes Sake cup"""
    
    # =========================================================================
    # Special Yaku Combinations (役に使うカードセット)
    # =========================================================================
    
    BOAR_DEER_BUTTERFLY: FrozenSet[Tuple[int, int]] = frozenset({
        (6, 1),   # Butterflies (June)
        (7, 1),   # Boar (July)
        (10, 1),  # Deer (October)
    })
    """猪鹿蝶 (Ino-Shika-Chō) - Boar, Deer, Butterflies"""
    
    FLOWER_SAKE: FrozenSet[Tuple[int, int]] = frozenset({
        (3, 1),   # Curtain (Cherry)
        (9, 1),   # Sake Cup
    })
    """花見酒 (Hanami-zake) - Flower Viewing Sake"""
    
    MOON_SAKE: FrozenSet[Tuple[int, int]] = frozenset({
        (8, 1),   # Moon
        (9, 1),   # Sake Cup
    })
    """月見酒 (Tsukimi-zake) - Moon Viewing Sake"""
    
    RED_RIBBON: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 2),   # Pine Ribbon
        (2, 2),   # Plum Ribbon
        (3, 2),   # Cherry Ribbon
    })
    """赤短 (Aka-tan) - Red Poetry Ribbons"""
    
    BLUE_RIBBON: FrozenSet[Tuple[int, int]] = frozenset({
        (6, 2),   # Peony Ribbon
        (9, 2),   # Chrysanthemum Ribbon
        (10, 2),  # Maple Ribbon
    })
    """青短 (Ao-tan) - Blue Ribbons"""
    
    RED_BLUE_RIBBON: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 2), (2, 2), (3, 2),   # Red Poetry
        (6, 2), (9, 2), (10, 2),  # Blue
    })
    """赤短・青短 - All special ribbons for combined yaku"""


class CardEncoder:
    """
    Utility class for encoding cards to various formats.
    
    This class provides methods to convert between different card representations
    used in the game logic and neural network inputs.
    """
    
    @staticmethod
    def to_multi_hot(cards: List[Card]) -> List[int]:
        """
        Convert a list of cards to a multi-hot encoding.
        
        Args:
            cards: List of Card objects
            
        Returns:
            List of 48 integers (0 or 1) representing card presence
            
        Example:
            >>> CardEncoder.to_multi_hot([Card(1, 1), Card(3, 1)])
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]  # length 48
        """
        encoding = [0] * TOTAL_CARDS
        for card in cards:
            encoding[card.to_index()] = 1
        return encoding
    
    @staticmethod
    def to_multi_hot_from_list(card_lists: List[List[int]]) -> List[int]:
        """
        Convert legacy [suit, rank] lists to multi-hot encoding.
        
        Args:
            card_lists: List of [suit, rank] lists
            
        Returns:
            List of 48 integers (0 or 1)
        """
        encoding = [0] * TOTAL_CARDS
        for card_list in card_lists:
            index = (card_list[0] - 1) * CARDS_PER_SUIT + (card_list[1] - 1)
            encoding[index] = 1
        return encoding
    
    @staticmethod
    def from_multi_hot(encoding: List[int]) -> List[Card]:
        """
        Convert a multi-hot encoding back to a list of cards.
        
        Args:
            encoding: List of 48 integers (0 or 1)
            
        Returns:
            List of Card objects
        """
        return [
            Card.from_index(i)
            for i, present in enumerate(encoding)
            if present == 1
        ]
    
    @staticmethod
    def cards_to_list_format(cards: List[Card]) -> List[List[int]]:
        """Convert Card objects to legacy [suit, rank] list format."""
        return [card.to_list() for card in cards]
    
    @staticmethod
    def list_format_to_cards(card_lists: List[List[int]]) -> List[Card]:
        """Convert legacy [suit, rank] lists to Card objects."""
        return [Card.from_list(cl) for cl in card_lists]


def create_full_deck() -> List[Card]:
    """
    Create a complete Hanafuda deck of 48 cards.
    
    Returns:
        List of all 48 cards in order (January rank 1-4, February rank 1-4, etc.)
    """
    return [
        Card(suit, rank)
        for suit in range(1, SUITS_COUNT + 1)
        for rank in range(1, CARDS_PER_SUIT + 1)
    ]


def classify_cards(cards: List[Card]) -> Tuple[List[Card], List[Card], List[Card], List[Card]]:
    """
    Classify cards into categories for display.
    
    Args:
        cards: List of cards to classify
        
    Returns:
        Tuple of (brights, seeds, ribbons, dross)
        
    Note:
        Sake cup (9,1) appears in both seeds and dross lists.
    """
    brights: List[Card] = []
    seeds: List[Card] = []
    ribbons: List[Card] = []
    dross: List[Card] = []
    
    for card in cards:
        card_tuple = card.to_tuple()
        
        if card.is_sake:
            # Sake cup counts as both seed and dross
            seeds.append(card)
            dross.append(card)
        elif card_tuple in CardSets.BRIGHT:
            brights.append(card)
        elif card_tuple in CardSets.SEED:
            seeds.append(card)
        elif card_tuple in CardSets.RIBBON:
            ribbons.append(card)
        else:
            dross.append(card)
    
    return brights, seeds, ribbons, dross
