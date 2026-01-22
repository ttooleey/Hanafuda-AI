"""
Yaku (winning hand) definitions and calculation for Koi-Koi.

This module defines all possible yaku combinations and provides
a calculator class to evaluate collected cards.

Yaku (役) are the winning hands in Koi-Koi. Players collect cards
by matching them, then form yaku combinations to score points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import FrozenSet, List, Tuple

from koikoi.core.card import CardSets, Card


class YakuType(IntEnum):
    """
    Enumeration of all possible yaku types.
    
    The integer values are used for identification and the order roughly
    corresponds to the rarity/value of each yaku.
    """
    # Light yaku (光系)
    FIVE_LIGHTS = 1        # 五光: All 5 bright cards (10 points)
    FOUR_LIGHTS = 2        # 四光: 4 brights without Rainman (8 points)
    RAINY_FOUR = 3         # 雨四光: 4 brights with Rainman (7 points)
    THREE_LIGHTS = 4       # 三光: 3 brights without Rainman (5 points)
    
    # Seed yaku (タネ系)
    BOAR_DEER_BUTTERFLY = 5   # 猪鹿蝶 (5 points)
    FLOWER_VIEWING = 6        # 花見酒 without koi-koi (1 point)
    FLOWER_VIEWING_KOIKOI = 7 # 花見酒 with koi-koi (3 points)
    MOON_VIEWING = 8          # 月見酒 without koi-koi (1 point)
    MOON_VIEWING_KOIKOI = 9   # 月見酒 with koi-koi (3 points)
    TANE = 10                 # タネ: 5+ seeds (n-4 points)
    
    # Ribbon yaku (短冊系)
    RED_BLUE_RIBBONS = 11  # 赤短・青短 (10 points)
    RED_RIBBONS = 12       # 赤短 (5 points)
    BLUE_RIBBONS = 13      # 青短 (5 points)
    TAN = 14               # 短冊: 5+ ribbons (n-4 points)
    
    # Dross yaku (カス系)
    KASU = 15              # カス: 10+ dross (n-9 points)
    
    # Bonus
    KOIKOI_BONUS = 16      # こいこいボーナス


@dataclass(frozen=True)
class Yaku:
    """
    Represents a completed yaku (winning hand).
    
    Attributes:
        yaku_type: The type of yaku
        name: Display name in English
        name_jp: Display name in Japanese
        base_points: Base point value
    """
    yaku_type: YakuType
    name: str
    name_jp: str
    base_points: int
    
    def to_tuple(self) -> Tuple[int, str, int]:
        """
        Convert to legacy tuple format.
        
        Returns:
            (type_id, name, points) tuple for backward compatibility
        """
        return (self.yaku_type.value, self.name, self.base_points)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.name_jp}): {self.base_points} pts"


# =============================================================================
# Yaku Definitions
# =============================================================================

YAKU_DEFINITIONS = {
    YakuType.FIVE_LIGHTS: Yaku(YakuType.FIVE_LIGHTS, "Five Lights", "五光", 10),
    YakuType.FOUR_LIGHTS: Yaku(YakuType.FOUR_LIGHTS, "Four Lights", "四光", 8),
    YakuType.RAINY_FOUR: Yaku(YakuType.RAINY_FOUR, "Rainy Four Lights", "雨四光", 7),
    YakuType.THREE_LIGHTS: Yaku(YakuType.THREE_LIGHTS, "Three Lights", "三光", 5),
    YakuType.BOAR_DEER_BUTTERFLY: Yaku(YakuType.BOAR_DEER_BUTTERFLY, "Boar-Deer-Butterfly", "猪鹿蝶", 5),
    YakuType.FLOWER_VIEWING: Yaku(YakuType.FLOWER_VIEWING, "Flower Viewing Sake", "花見酒", 1),
    YakuType.FLOWER_VIEWING_KOIKOI: Yaku(YakuType.FLOWER_VIEWING_KOIKOI, "Flower Viewing Sake", "花見酒", 3),
    YakuType.MOON_VIEWING: Yaku(YakuType.MOON_VIEWING, "Moon Viewing Sake", "月見酒", 1),
    YakuType.MOON_VIEWING_KOIKOI: Yaku(YakuType.MOON_VIEWING_KOIKOI, "Moon Viewing Sake", "月見酒", 3),
    YakuType.RED_BLUE_RIBBONS: Yaku(YakuType.RED_BLUE_RIBBONS, "Red & Blue Ribbons", "赤短・青短", 10),
    YakuType.RED_RIBBONS: Yaku(YakuType.RED_RIBBONS, "Red Ribbons", "赤短", 5),
    YakuType.BLUE_RIBBONS: Yaku(YakuType.BLUE_RIBBONS, "Blue Ribbons", "青短", 5),
}


class YakuCalculator:
    """
    Calculator for determining yaku from collected cards.
    
    This class analyzes a player's collected pile to find all
    completed yaku and calculate total points.
    
    Example:
        >>> calculator = YakuCalculator()
        >>> pile = [Card(1, 1), Card(3, 1), Card(8, 1)]  # Three bright cards
        >>> yaku_list = calculator.calculate(pile, koikoi_count=0)
        >>> for yaku in yaku_list:
        ...     print(f"{yaku.name}: {yaku.base_points} points")
        Three Lights: 5 points
    """
    
    def calculate(self, pile: List[Card], koikoi_count: int) -> List[Yaku]:
        """
        Calculate all yaku from a player's collected pile.
        
        Args:
            pile: List of collected cards
            koikoi_count: Number of times player has called koi-koi
            
        Returns:
            List of completed Yaku objects
        """
        yaku_list: List[Yaku] = []
        pile_set = frozenset(card.to_tuple() for card in pile)
        
        # Check each category
        yaku_list.extend(self._check_lights(pile_set))
        yaku_list.extend(self._check_seeds(pile_set, koikoi_count))
        yaku_list.extend(self._check_ribbons(pile_set))
        yaku_list.extend(self._check_dross(pile_set))
        
        # Add koi-koi bonus
        if koikoi_count > 0:
            yaku_list.append(Yaku(
                YakuType.KOIKOI_BONUS,
                "Koi-Koi",
                "こいこい",
                koikoi_count
            ))
        
        return yaku_list
    
    def calculate_points(self, pile: List[Card], koikoi_count: int) -> int:
        """
        Calculate total points from yaku.
        
        Scoring rules:
        - Sum base points of all yaku (excluding Koi-Koi bonus)
        - If koi-koi count <= 3: add koi-koi count as bonus
        - If koi-koi count > 3: multiply total by (koi-koi count - 2)
        
        Args:
            pile: List of collected cards
            koikoi_count: Number of times player has called koi-koi
            
        Returns:
            Total point value
        """
        yaku_list = self.calculate(pile, koikoi_count)
        
        # Sum base points excluding koi-koi bonus
        base_points = sum(
            yaku.base_points
            for yaku in yaku_list
            if yaku.yaku_type != YakuType.KOIKOI_BONUS
        )
        
        # Apply koi-koi scoring
        if koikoi_count <= 3:
            return base_points + koikoi_count
        else:
            return base_points * (koikoi_count - 2)
    
    def _check_lights(self, pile_set: FrozenSet[Tuple[int, int]]) -> List[Yaku]:
        """Check for light-based yaku (光系)."""
        yaku_list: List[Yaku] = []
        light_count = len(pile_set & CardSets.BRIGHT)
        has_rainman = (11, 1) in pile_set
        
        if light_count == 5:
            # 五光: All 5 brights
            yaku_list.append(YAKU_DEFINITIONS[YakuType.FIVE_LIGHTS])
        elif light_count == 4 and not has_rainman:
            # 四光: 4 brights without Rainman
            yaku_list.append(YAKU_DEFINITIONS[YakuType.FOUR_LIGHTS])
        elif light_count == 4 and has_rainman:
            # 雨四光: 4 brights including Rainman
            yaku_list.append(YAKU_DEFINITIONS[YakuType.RAINY_FOUR])
        elif light_count == 3 and not has_rainman:
            # 三光: 3 brights without Rainman
            yaku_list.append(YAKU_DEFINITIONS[YakuType.THREE_LIGHTS])
        
        return yaku_list
    
    def _check_seeds(
        self,
        pile_set: FrozenSet[Tuple[int, int]],
        koikoi_count: int
    ) -> List[Yaku]:
        """Check for seed-based yaku (タネ系)."""
        yaku_list: List[Yaku] = []
        seed_count = len(pile_set & CardSets.SEED)
        
        # 猪鹿蝶: Boar-Deer-Butterfly
        if CardSets.BOAR_DEER_BUTTERFLY.issubset(pile_set):
            yaku_list.append(YAKU_DEFINITIONS[YakuType.BOAR_DEER_BUTTERFLY])
        
        # 花見酒: Flower Viewing Sake
        if CardSets.FLOWER_SAKE.issubset(pile_set):
            if koikoi_count == 0:
                yaku_list.append(YAKU_DEFINITIONS[YakuType.FLOWER_VIEWING])
            else:
                yaku_list.append(YAKU_DEFINITIONS[YakuType.FLOWER_VIEWING_KOIKOI])
        
        # 月見酒: Moon Viewing Sake
        if CardSets.MOON_SAKE.issubset(pile_set):
            if koikoi_count == 0:
                yaku_list.append(YAKU_DEFINITIONS[YakuType.MOON_VIEWING])
            else:
                yaku_list.append(YAKU_DEFINITIONS[YakuType.MOON_VIEWING_KOIKOI])
        
        # タネ: 5+ seeds
        if seed_count >= 5:
            yaku_list.append(Yaku(YakuType.TANE, "Tane", "タネ", seed_count - 4))
        
        return yaku_list
    
    def _check_ribbons(self, pile_set: FrozenSet[Tuple[int, int]]) -> List[Yaku]:
        """Check for ribbon-based yaku (短冊系)."""
        yaku_list: List[Yaku] = []
        ribbon_count = len(pile_set & CardSets.RIBBON)
        
        # 赤短・青短: Red & Blue Ribbons together
        if CardSets.RED_BLUE_RIBBON.issubset(pile_set):
            yaku_list.append(YAKU_DEFINITIONS[YakuType.RED_BLUE_RIBBONS])
        
        # 赤短: Red Poetry Ribbons
        if CardSets.RED_RIBBON.issubset(pile_set):
            yaku_list.append(YAKU_DEFINITIONS[YakuType.RED_RIBBONS])
        
        # 青短: Blue Ribbons
        if CardSets.BLUE_RIBBON.issubset(pile_set):
            yaku_list.append(YAKU_DEFINITIONS[YakuType.BLUE_RIBBONS])
        
        # 短冊: 5+ ribbons
        if ribbon_count >= 5:
            yaku_list.append(Yaku(YakuType.TAN, "Tan", "短冊", ribbon_count - 4))
        
        return yaku_list
    
    def _check_dross(self, pile_set: FrozenSet[Tuple[int, int]]) -> List[Yaku]:
        """Check for dross-based yaku (カス系)."""
        yaku_list: List[Yaku] = []
        dross_count = len(pile_set & CardSets.DROSS)
        
        # カス: 10+ dross
        if dross_count >= 10:
            yaku_list.append(Yaku(YakuType.KASU, "Kasu", "カス", dross_count - 9))
        
        return yaku_list
