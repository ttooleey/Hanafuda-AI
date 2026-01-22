"""
Helper utilities for Koi-Koi AI.

This module provides common utility functions used across
the codebase, including:
- Game record I/O
- Action encoding/decoding
- Data format conversions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from koikoi.core.constants import Action, CardAction, KoiKoiAction


def action_to_index(action: Action) -> Optional[int]:
    """
    Convert action to integer index.
    
    Handles three action formats:
    - Card actions: [suit, rank] -> (suit-1)*4 + (rank-1)
    - Koi-koi decisions: True/False -> 1/0
    - None -> None
    
    Args:
        action: Action in game format
        
    Returns:
        Integer index or None
        
    Examples:
        >>> action_to_index([1, 1])  # First card (suit 1, rank 1)
        0
        >>> action_to_index([12, 4])  # Last card (suit 12, rank 4)
        47
        >>> action_to_index(True)  # Continue (koi-koi)
        1
        >>> action_to_index(False)  # Stop
        0
    """
    if action is None:
        return None
    if action in [False, True]:
        return int(action)
    if isinstance(action, (list, tuple)) and len(action) == 2:
        suit, rank = action
        return 4 * (suit - 1) + (rank - 1)
    raise ValueError(f"Invalid action format: {action}")


def index_to_action(index: int, action_type: str = 'card') -> Union[CardAction, KoiKoiAction]:
    """
    Convert integer index back to action format.
    
    Args:
        index: Integer index (0-47 for cards, 0-1 for koi-koi)
        action_type: 'card' or 'koikoi'
        
    Returns:
        Action in game format
        
    Examples:
        >>> index_to_action(0, 'card')
        [1, 1]
        >>> index_to_action(47, 'card')
        [12, 4]
        >>> index_to_action(1, 'koikoi')
        True
    """
    if action_type == 'koikoi':
        return bool(index)
    elif action_type == 'card':
        suit = index // 4 + 1
        rank = index % 4 + 1
        return [suit, rank]
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def load_game_record(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a game record from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary containing game record
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_game_record(record: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a game record to JSON file.
    
    Args:
        record: Game record dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def card_to_string(suit: int, rank: int) -> str:
    """
    Convert card indices to human-readable string.
    
    Args:
        suit: Suit number (1-12 for months)
        rank: Rank within suit (1-4)
        
    Returns:
        Human-readable card name
        
    Example:
        >>> card_to_string(1, 1)
        '松に鶴'
    """
    month_names = [
        '', '松', '梅', '桜', '藤', '菖蒲', '牡丹',
        '萩', '芒', '菊', '紅葉', '柳', '桐'
    ]
    
    # Special cards (光札, タネ札) have unique names
    special_cards = {
        (1, 1): '松に鶴',
        (3, 1): '桜に幕',
        (8, 1): '芒に月',
        (11, 1): '柳に小野道風',
        (12, 1): '桐に鳳凰',
        (2, 1): '梅に鶯',
        (4, 1): '藤に不如帰',
        (5, 1): '菖蒲に八橋',
        (6, 1): '牡丹に蝶',
        (7, 1): '萩に猪',
        (8, 2): '芒に雁',
        (9, 1): '菊に盃',
        (10, 1): '紅葉に鹿',
        (11, 2): '柳に燕',
    }
    
    if (suit, rank) in special_cards:
        return special_cards[(suit, rank)]
    
    month = month_names[suit]
    return f'{month}の{rank}番札'


def format_yaku_list(yakus: List[Tuple[str, int]]) -> str:
    """
    Format yaku list for display.
    
    Args:
        yakus: List of (yaku_name, points) tuples
        
    Returns:
        Formatted string
        
    Example:
        >>> format_yaku_list([('三光', 5), ('赤短', 5)])
        '三光 (5点), 赤短 (5点)'
    """
    return ', '.join(f'{name} ({pts}点)' for name, pts in yakus)


def mask_to_valid_actions(mask: List[int], action_type: str = 'card') -> List[Any]:
    """
    Convert action mask to list of valid actions.
    
    Args:
        mask: Binary mask where 1 indicates valid action
        action_type: 'card' (48 actions) or 'koikoi' (2 actions)
        
    Returns:
        List of valid actions in game format
    """
    valid = []
    for i, is_valid in enumerate(mask):
        if is_valid:
            valid.append(index_to_action(i, action_type))
    return valid
