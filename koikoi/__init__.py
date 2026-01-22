"""
KoiKoi AI - A Transformer-based AI for playing Koi-Koi Hanafuda card games.

This package provides:
- Core game logic (rules, cards, yaku)
- AI agents with different strategies
- Training utilities for reinforcement learning
- GUI for playing against AI

Reference:
    S. Guan, J. Wang, R. Zhu, J. Qian and Z. Wei,
    "Learning to Play Koi-Koi Hanafuda Card Games with Transformers,"
    IEEE Transactions on Artificial Intelligence, vol. 4, no. 6, pp. 1449-1460, 2023.

Usage:
    # Play against AI with GUI
    python -m koikoi.play --ai RL-Point --name YourName
"""

from koikoi.core.game_state import KoiKoiGameState
from koikoi.core.round_state import KoiKoiRoundState
from koikoi.core.card import Card, CardCategory, CardSets
from koikoi.core.yaku import Yaku, YakuType, YakuCalculator
from koikoi.ai.agent import KoiKoiAgent

__version__ = "0.2.0"
__all__ = [
    # Core
    "KoiKoiGameState",
    "KoiKoiRoundState",
    "Card",
    "CardCategory",
    "CardSets",
    "Yaku",
    "YakuType",
    "YakuCalculator",
    # AI
    "KoiKoiAgent",
]
